"""Meta-avaliação das métricas de qualidade de resumo.

Pergunta respondida: qual métrica (barata ou moderna) concorda mais com a nota humana
de qualidade (gt_qualidade_resumo) e quanto custa por 1.000 resumos?

Para cada métrica: Spearman (IC bootstrap), Kendall tau-b, e AUC para detectar
resumos ruins (nota humana <= 2) — o uso prático mais comum em produção (triagem).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import roc_auc_score

from kpibench.config import BenchmarkConfig
from kpibench.data.loader import Dataset
from kpibench.metrics.stats import bootstrap_indices, percentile_ci
from kpibench.metrics.summary import lexical, neural
from kpibench.metrics.summary.base import MetricResult

BAD_THRESHOLD = 2


def compute_summary_metrics(cfg: BenchmarkConfig, ds: Dataset, preds: pd.DataFrame,
                            claims: pd.DataFrame | None) -> list[MetricResult]:
    sm = cfg.summary_metrics
    eff, mod = set(sm.efficient), set(sm.modern)
    ids = ds.ids()
    summaries = ds.df[ds.summary_col].astype(str).tolist()
    sources = ds.df[ds.transcript_col].fillna("").astype(str).tolist() if ds.has_transcript else None

    if ds.has_reference:
        refs, ref_kind = ds.df[ds.reference_col].fillna("").astype(str).tolist(), "referencia"
    elif sources is not None:
        refs, ref_kind = sources, "pseudo_referencia(fonte)"
    else:
        refs, ref_kind = None, None

    results: list[MetricResult] = []
    # ---------------- eficientes
    if refs is not None:
        results += lexical.reference_metrics(summaries, refs, ref_kind, eff)
    if sources is not None:
        results += lexical.source_metrics(summaries, sources, eff)

    # ---------------- modernas: modelos locais
    if refs is not None and "bertscore" in mod:
        results.append(neural.bertscore(summaries, refs, ref_kind, sm.bertscore_model, sm.device))
    if refs is not None and "embedding_cosine" in mod:
        results.append(neural.embedding_cosine(summaries, refs, ref_kind, sm.embedding_model, sm.device))
    if sources is not None and "summac_zs" in mod:
        results.append(neural.summac_zs(summaries, sources, sm.nli_model, sm.device))

    # ---------------- modernas: LLM-juiz
    if claims is not None and not claims.empty:
        cm = claims.set_index("call_id")
        vals = np.array([cm["claim_faithfulness"].get(i, np.nan) for i in ids], dtype=float)
        results.append(MetricResult(
            f"claim_faithfulness[{sm.claim_judge_model}]", "llm_juiz", "moderna", "fonte", vals,
            compute="API", ms_per_item=float(cm["latency_s"].mean() * 1000),
            usd_per_1k=float(cm["cost_usd"].mean() * 1000),
            note="afirmações atômicas verificadas contra a transcrição (estilo FActScore/QAG)"))

    if "geval" in mod and preds is not None and not preds.empty:
        q = preds[preds["kpi"] == "qualidade_resumo"]
        for m, g in q.groupby("model"):
            gm = g.set_index("call_id")
            vals = np.array([gm["value"].get(i, np.nan) if gm["valid"].get(i, False) else np.nan for i in ids],
                            dtype=float)
            results.append(MetricResult(
                f"geval[{m}]", "llm_juiz", "moderna", "fonte" if ds.has_transcript else "sem_referencia",
                vals, compute="API", ms_per_item=float(g["latency_s"].mean() * 1000),
                usd_per_1k=float(g["cost_usd"].mean() * 1000),
                note="rubrica CoT + preenchimento de formulário (prompt do KPI qualidade_resumo)"))
    return results


def meta_evaluate(cfg: BenchmarkConfig, ds: Dataset, results: list[MetricResult]) -> pd.DataFrame:
    gt_col = cfg.kpi_registry["qualidade_resumo"].gt_column if "qualidade_resumo" in cfg.kpi_registry else None
    has_gt = gt_col is not None and gt_col in ds.df.columns
    human = pd.to_numeric(ds.df[gt_col], errors="coerce").to_numpy(float) if has_gt else None
    idx = bootstrap_indices(len(ds.df), cfg.statistics.n_bootstrap, cfg.run.seed + 1)

    rows = []
    for r in results:
        row = {"metric": r.name, "grupo": r.group, "familia": r.family, "referencia": r.reference,
               "compute": r.compute, "ms_por_item": r.ms_per_item, "usd_por_1k": r.usd_per_1k,
               "nota": r.note, "status": "ok" if r.skipped_reason is None else f"pulada: {r.skipped_reason}"}
        if r.skipped_reason is None and has_gt:
            x = r.oriented
            mask = np.isfinite(x) & np.isfinite(human)
            row["cobertura"] = float(mask.mean())
            if mask.sum() >= 5 and np.unique(x[mask]).size > 1:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    row["spearman"] = float(spearmanr(x[mask], human[mask]).statistic)
                    row["kendall_tau_b"] = float(kendalltau(x[mask], human[mask]).statistic)

                    def rho(ix, x=x, h=human):
                        m = np.isfinite(x[ix]) & np.isfinite(h[ix])
                        xs, hs = x[ix][m], h[ix][m]
                        if len(xs) < 5 or np.unique(xs).size < 2 or np.unique(hs).size < 2:
                            return np.nan
                        return spearmanr(xs, hs).statistic
                    dist = np.array([rho(ix) for ix in idx])
                lo, hi = percentile_ci(dist, cfg.statistics.ci)
                row["spearman_ci_low"], row["spearman_ci_high"] = lo, hi
                bad = (human[mask] <= BAD_THRESHOLD).astype(int)
                if 0 < bad.sum() < len(bad):
                    row["auc_detecta_ruim"] = float(roc_auc_score(bad, -x[mask]))
        rows.append(row)

    df = pd.DataFrame(rows)
    if "spearman" in df.columns:
        df = df.sort_values("spearman", ascending=False, na_position="last")
    return df.reset_index(drop=True)


def per_item_table(ds: Dataset, results: list[MetricResult], gt_col: str | None) -> pd.DataFrame:
    out = pd.DataFrame({"call_id": ds.ids()})
    if gt_col and gt_col in ds.df.columns:
        out["nota_humana"] = ds.df[gt_col].to_numpy()
    for r in results:
        if r.skipped_reason is None:
            out[r.name] = r.values
    return out
