"""Orquestração: extração -> juízes -> avaliação -> custo -> roteamento -> relatório."""
from __future__ import annotations

import json
import platform
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from kpibench import __version__
from kpibench.analysis.cost import cost_latency_table, evaluation_overhead
from kpibench.analysis.routing import routing
from kpibench.config import BenchmarkConfig
from kpibench.data.loader import Dataset, load_dataset
from kpibench.extraction.runner import (LLMExecutor, run_claim_faithfulness, run_kpi_extraction,
                                        run_reasoning_judge)
from kpibench.metrics.kpi_eval import BASELINE_NAME, evaluate_kpi
from kpibench.metrics.summary.meta_eval import compute_summary_metrics, meta_evaluate, per_item_table
from kpibench.prompts.renderer import PromptRenderer
from kpibench.reporting import figures
from kpibench.reporting.report import build_report


def make_run_dir(cfg: BenchmarkConfig) -> Path:
    out = cfg.resolve(cfg.run.output_dir) / f"{cfg.run.name}_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _baseline_text_rows(cfg: BenchmarkConfig, ds: Dataset) -> pd.DataFrame:
    """Justificativas do modelo em produção (prod_<kpi>) entram no juiz como mais um candidato."""
    rows = []
    if not ds.baseline_prefix:
        return pd.DataFrame()
    for k in cfg.selected_kpis:
        col = f"{ds.baseline_prefix}{k.name}"
        if k.type == "text" and col in ds.df.columns:
            for cid, v in zip(ds.ids(), ds.df[col]):
                ok = isinstance(v, str) and bool(v.strip())
                rows.append({"model": BASELINE_NAME, "kpi": k.name, "call_id": cid, "value": v if ok else None,
                             "valid": ok})
    return pd.DataFrame(rows)


def run(cfg: BenchmarkConfig) -> Path:
    ds = load_dataset(cfg)
    for w in ds.warnings:
        print(f"[aviso] {w}", file=sys.stderr)
    out = make_run_dir(cfg)
    renderer = PromptRenderer(cfg.root / "prompts")
    ex = LLMExecutor(cfg, ds)

    preds = run_kpi_extraction(cfg, ds, renderer, ex)
    preds.to_csv(out / "predictions.csv", index=False)

    base_txt = _baseline_text_rows(cfg, ds)
    judge_input = pd.concat([preds, base_txt], ignore_index=True) if not base_txt.empty else preds
    judge = run_reasoning_judge(cfg, ds, judge_input, renderer, ex)
    judge.to_csv(out / "judge_scores.csv", index=False)

    claims = run_claim_faithfulness(cfg, ds, renderer, ex)
    if not claims.empty:
        claims.to_csv(out / "claim_faithfulness.csv", index=False)

    evaluate(cfg, ds, preds, judge, claims, out, renderer)
    return out


def evaluate(cfg: BenchmarkConfig, ds: Dataset, preds: pd.DataFrame, judge: pd.DataFrame | None,
             claims: pd.DataFrame | None, out: Path, renderer: PromptRenderer | None = None) -> Path:
    renderer = renderer or PromptRenderer(cfg.root / "prompts")
    fig_dir = out

    evals = {}
    for kpi in cfg.selected_kpis:
        ev = evaluate_kpi(cfg, ds, kpi, preds, judge)
        if ev is not None:
            evals[kpi.name] = ev
    if not evals:
        raise RuntimeError("nenhum KPI pôde ser avaliado (verifique as colunas gt_*)")

    costs = cost_latency_table(cfg, preds)
    full, rec, plans = routing(cfg, evals, costs)

    sm_results = compute_summary_metrics(cfg, ds, preds, claims)
    meta = meta_evaluate(cfg, ds, sm_results) if sm_results else pd.DataFrame()
    gt_q = cfg.kpi_registry["qualidade_resumo"].gt_column if "qualidade_resumo" in cfg.kpi_registry else None

    # ---------------- artefatos tabulares
    full.to_csv(out / "metricas_por_kpi_modelo.csv", index=False)
    rec.to_csv(out / "recomendacao_roteamento.csv", index=False)
    plans.to_csv(out / "planos_custo_mensal.csv", index=False)
    costs.to_csv(out / "custos_latencia.csv", index=False)
    if not meta.empty:
        meta.to_csv(out / "meta_avaliacao_metricas_resumo.csv", index=False)
        per_item_table(ds, sm_results, gt_q).to_csv(out / "metricas_resumo_por_item.csv", index=False)
    pcs = [ev.per_class for ev in evals.values() if ev.per_class is not None]
    if pcs:
        pd.concat(pcs).to_csv(out / "f1_por_classe.csv", index=False)

    # ---------------- figuras + relatório
    figs = {
        "pareto": figures.pareto_grid(full, rec, fig_dir),
        "heatmap": figures.heatmap(full[full["model"] != BASELINE_NAME], rec, fig_dir),
        "meta": figures.summary_meta_eval(meta, fig_dir) if not meta.empty else None,
    }
    hashes = {k.name: renderer.template_hash(f"kpis/{k.prompt}") for k in cfg.selected_kpis}
    report = build_report(cfg, ds, evals, full, rec, plans, costs, meta, hashes,
                          evaluation_overhead(judge, claims), figs, out)

    manifest = {
        "kpibench_version": __version__, "python": platform.python_version(),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "n_records": len(ds.df), "models": cfg.models, "kpis": list(evals),
        "prompt_hashes": hashes, "judge": cfg.judge.model if cfg.judge.enabled else None,
        "simulated": bool(costs["simulated"].any()),
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    cfg_dump = cfg.model_dump(mode="json", exclude={"model_registry", "kpi_registry", "root"})
    (out / "config_snapshot.yaml").write_text(yaml.safe_dump(cfg_dump, allow_unicode=True, sort_keys=False))
    print(f"\nRelatório: {report}", file=sys.stderr)
    return report


def evaluate_existing(cfg: BenchmarkConfig, run_dir: Path) -> Path:
    """Reavalia uma rodada salva (ex.: mudou margem, bootstrap ou projeção) sem chamar APIs."""
    ds = load_dataset(cfg)
    preds = pd.read_csv(run_dir / "predictions.csv", dtype={"call_id": str})
    judge_p = run_dir / "judge_scores.csv"
    claims_p = run_dir / "claim_faithfulness.csv"
    judge = pd.read_csv(judge_p, dtype={"call_id": str}) if judge_p.exists() and judge_p.stat().st_size > 1 else None
    claims = pd.read_csv(claims_p, dtype={"call_id": str}) if claims_p.exists() else None
    return evaluate(cfg, ds, preds, judge, claims, run_dir)
