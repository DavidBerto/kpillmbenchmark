"""Extração consolidada: UMA chamada de LLM por ligação, sobre a TRANSCRIÇÃO COMPLETA, devolvendo
todos os KPIs num único JSON. Mede a redução de custo/tokens e a variação de qualidade frente ao
fluxo atual (uma chamada por KPI sobre o resumo). Usado pelo notebook 04.

Abordagens comparadas (mesma referência, mesmas ligações):
  por_kpi_resumo           fluxo atual: N chamadas por ligação (1 por KPI), entrada = resumo
  consolidado_transcricao  fluxo novo:  1 chamada por ligação, entrada = transcrição completa,
                           saída = {"kpis": {"<kpi>": {"justificativa": ..., "valor": ...}, ...}}
  producao_atual           KPIs já gravados em produção, só nos KPIs de domínio comparável

Prompt consolidado (prompts/consolidado/todos_kpis.md): as rubricas de cada KPI vêm dos próprios
templates por KPI (prompts/kpis/<kpi>.md, sem o bloco de formato), então os critérios são os
mesmos dos dois fluxos; muda só a fonte (transcrição) e o envelope (um JSON com todos os KPIs).

Referência (o que conta como "certo"):
  "arquivo:<csv>"        rótulos auditados (call_id + colunas gt) — a opção sem viés;
  modelo + "consolidado" modelo forte lê a transcrição com o prompt consolidado (padrão). Vê toda
                         a informação da ligação, mas compartilha o prompt do fluxo novo: tende a
                         favorecê-lo;
  modelo + "por_kpi"     modelo forte com os prompts por KPI sobre o resumo: tende a favorecer o
                         fluxo atual (não vê o que o resumo perdeu).
"""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from kpibench.analysis.cost_effectiveness import normalize_quality, quality_bounds
from kpibench.config import BenchmarkConfig, KPISpec
from kpibench.data.loader import Dataset, load_dataset
from kpibench.extraction.parser import ParsedOutput, parse_all_kpis_output
from kpibench.extraction.runner import (LLMExecutor, _run_parallel, _usage_cols, kpi_prompt_vars,
                                        run_kpi_extraction, run_reasoning_judge)
from kpibench.llm.pricing import projected_cost_per_call
from kpibench.llm.tokens import estimate_tokens
from kpibench.metrics.kpi_eval import BASELINE_NAME, KPIEvaluation, evaluate_kpi
from kpibench.metrics.stats import percentile_ci
from kpibench.prompts.renderer import PromptRenderer, RenderedPrompt

TEMPLATE = "consolidado/todos_kpis.md"
PER_KPI = "por_kpi_resumo"
SINGLE = "consolidado_transcricao"
APPROACH_LABELS = {PER_KPI: "Atual: 1 chamada por KPI (resumo)",
                   SINGLE: "Novo: 1 chamada por ligação (transcrição)",
                   BASELINE_NAME: "Produção atual"}
TOKEN_COLS = ["input_tokens", "cached_tokens", "cache_write_tokens", "output_tokens", "reasoning_tokens"]

# saída assumida na estimativa (antes de executar), em tokens
OUT_PER_KPI = {"ordinal": 60, "categorical": 60, "text": 70}
OUT_SUMMARY_QUALITY = 250
REASONING_OVERHEAD = 300


# =============================================================================== prompt consolidado
def kpi_rubric(renderer: PromptRenderer, kpi: KPISpec) -> str:
    """Rubrica do KPI = SYSTEM do template por KPI, sem as regras gerais e sem o bloco de formato."""
    p = renderer.render(f"kpis/{kpi.prompt}", resumo="", transcricao="", has_transcript=True,
                        labels=kpi.labels or {}, scale=kpi.scale.model_dump() if kpi.scale else None)
    base = renderer.env.get_template("_base_system.md").render().strip()
    body = p.system.replace(base, "", 1).strip()
    return body.split("Formato de saída")[0].strip()


def _value_spec(kpi: KPISpec) -> tuple[str, str]:
    if kpi.type == "ordinal":
        rng = f"{kpi.scale.min}-{kpi.scale.max}"
        return f"inteiro de {kpi.scale.min} a {kpi.scale.max}", f"<inteiro {rng}>"
    if kpi.type == "categorical":
        labels = list(kpi.labels or {})
        return "exatamente um de: " + ", ".join(f'"{x}"' for x in labels), '"<' + "|".join(labels) + '>"'
    return "texto livre de até 40 palavras (o próprio indicador)", '"<até 40 palavras>"'


def kpi_block(renderer: PromptRenderer, kpi: KPISpec, justificativas: bool = True) -> dict:
    spec, example_value = _value_spec(kpi)
    parts = []
    if kpi.name == "qualidade_resumo":
        parts.append('"alucinacoes": ["<afirmação sem suporte>"]')
    if justificativas and kpi.type != "text":
        parts.append('"justificativa": "<texto curto>"')
    parts.append(f'"valor": {example_value}')
    return {"nome": kpi.name, "rubrica": kpi_rubric(renderer, kpi), "valor_spec": spec,
            "exemplo": "{" + ", ".join(parts) + "}"}


def prompt_hash(renderer: PromptRenderer, kpis: list[KPISpec]) -> str:
    """sha256 do template consolidado + templates por KPI usados nas rubricas + regras gerais."""
    h = hashlib.sha256()
    for rel in [TEMPLATE, "_base_system.md"] + [f"kpis/{k.prompt}" for k in kpis]:
        h.update((renderer.dir / rel).read_bytes())
    return h.hexdigest()[:12]


def render_single_call(renderer: PromptRenderer, kpis: list[KPISpec], transcricao: str, resumo: str = "",
                       max_palavras: int = 25, justificativas: bool = True) -> RenderedPrompt:
    tem_resumo = any(k.name == "qualidade_resumo" for k in kpis) and bool(str(resumo).strip())
    p = renderer.render(TEMPLATE, blocos=[kpi_block(renderer, k, justificativas) for k in kpis],
                        transcricao=str(transcricao), resumo=str(resumo), tem_resumo=tem_resumo,
                        max_palavras=max_palavras, com_justificativa=justificativas)
    return replace(p, template_hash=prompt_hash(renderer, kpis))


def single_call_kpis(cfg: BenchmarkConfig, include_summary_quality: bool) -> list[KPISpec]:
    return [k for k in cfg.selected_kpis if include_summary_quality or k.name != "qualidade_resumo"]


# =============================================================================== execução
def config_with_max_tokens(cfg: BenchmarkConfig, model_id: str, max_tokens: int | None) -> BenchmarkConfig:
    """Cópia do cfg com `max_tokens` maior para o modelo: a saída consolidada traz todos os KPIs."""
    c = cfg.model_copy(deep=True)
    if max_tokens:
        spec = c.model_registry[model_id]
        c.model_registry[model_id] = spec.model_copy(update={"params": {**spec.params, "max_tokens": max_tokens}})
    return c


def _valid(v) -> bool:
    return v is not None and not (isinstance(v, float) and np.isnan(v))


def run_single_call(cfg: BenchmarkConfig, ds: Dataset, model_id: str, renderer: PromptRenderer,
                    ex: LLMExecutor, kpis: list[KPISpec], *, label: str = SINGLE, max_palavras: int = 25,
                    justificativas: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Uma chamada por ligação. Retorna (por_ligacao, predicoes_longas).

    `predicoes_longas` tem o mesmo esquema de `run_kpi_extraction` (uma linha por KPI), com tokens e
    custo da chamada rateados igualmente entre os KPIs — some por ligação para o custo real.
    """
    if not ds.transcript_col:
        raise ValueError("a extração consolidada exige a coluna de transcrição")
    spec = cfg.model(model_id)
    jobs = []
    for cid in ds.ids():
        rec = ds.record(cid)

        def job(cid=cid, rec=rec) -> dict:
            transcript = rec.get(ds.transcript_col)
            transcript = "" if not _valid(transcript) else str(transcript)
            prompt = render_single_call(renderer, kpis, transcript, rec.get(ds.summary_col) or "",
                                        max_palavras, justificativas)
            res = ex.call(model_id, prompt, {"task": "kpi_all", "kpi_specs": kpis, "call_id": cid, "record": rec})
            if res.response is not None:
                obj, parsed = parse_all_kpis_output(res.response.text, kpis)
            else:
                obj, parsed = None, {k.name: ParsedOutput(None, False, res.error) for k in kpis}
            usage = _usage_cols(spec, res)
            finish = res.response.finish_reason if res.response else None
            n_ok = sum(p.valid for p in parsed.values())
            saida = {"call_id": cid, "modelo": model_id,
                     "kpis": {k.name: {"valor": parsed[k.name].value, "valido": parsed[k.name].valid,
                                       **({"justificativa": parsed[k.name].extras.get("justificativa")}
                                          if parsed[k.name].extras.get("justificativa") else {}),
                                       **({"erro": parsed[k.name].error} if parsed[k.name].error else {})}
                              for k in kpis}}
            call_row = {"abordagem": label, "modelo_llm": model_id, "call_id": cid, "chamadas_llm": 1,
                        "json_valido": obj is not None, "kpis_validos": n_ok, "n_kpis": len(kpis),
                        "finish_reason": finish, "truncado": finish == "length", "erro": res.error,
                        "prompt_hash": prompt.template_hash, "prefixo_estatico_tokens": prompt.static_prefix_tokens,
                        "conteudo_variavel_tokens": estimate_tokens(prompt.user),
                        "transcricao_chars": len(transcript), **usage,
                        "latencia_seq_s": usage["latency_s"], "latencia_paralela_s": usage["latency_s"],
                        "saida_json": json.dumps(saida, ensure_ascii=False)}
            share = {c: usage[c] / len(kpis) for c in TOKEN_COLS + ["cost_usd"]}
            preds = [{"model": label, "kpi": k.name, "call_id": cid, "value": parsed[k.name].value,
                      "valid": parsed[k.name].valid, "error": res.error or parsed[k.name].error,
                      "raw_output": None,
                      "extras": json.dumps(parsed[k.name].extras, ensure_ascii=False) if parsed[k.name].extras else None,
                      "prompt_hash": prompt.template_hash, "static_prefix_tokens": prompt.static_prefix_tokens,
                      **usage, **share, "modelo_llm": model_id} for k in kpis]
            return {"call": call_row, "preds": preds}
        jobs.append(job)
    print(f"Extração consolidada ({model_id}): {len(jobs)} chamadas (1 por ligação, {len(kpis)} KPIs cada)",
          file=sys.stderr)
    out = _run_parallel(jobs, cfg.run.max_workers, "consolidado")
    calls = pd.DataFrame([o["call"] for o in out])
    preds = pd.DataFrame([p for o in out for p in o["preds"]])
    return calls, preds


def per_call_from_per_kpi(preds: pd.DataFrame, ds: Dataset, label: str = PER_KPI) -> pd.DataFrame:
    """Agrega as N chamadas por KPI do fluxo atual em uma linha por ligação (mesmo esquema do consolidado)."""
    g = preds.groupby("call_id")
    out = pd.DataFrame({
        "abordagem": label, "modelo_llm": g["model"].first(), "chamadas_llm": g.size(),
        "json_valido": g["valid"].all(), "kpis_validos": g["valid"].sum(), "n_kpis": g.size(),
        "finish_reason": None, "truncado": False, "erro": g["error"].apply(lambda s: "; ".join(s.dropna().astype(str))[:300] or None),
        "prompt_hash": g["prompt_hash"].first(), "prefixo_estatico_tokens": g["static_prefix_tokens"].sum(),
        **{c: g[c].sum() for c in TOKEN_COLS + ["cost_usd"]},
        "latency_s": g["latency_s"].sum(), "latencia_seq_s": g["latency_s"].sum(),
        "latencia_paralela_s": g["latency_s"].max(),
        "simulated": g["simulated"].any(), "from_cache": g["from_cache"].all(),
    }).reset_index()
    total_in = out["input_tokens"] + out["cached_tokens"] + out["cache_write_tokens"]
    out["conteudo_variavel_tokens"] = (total_in - out["prefixo_estatico_tokens"]).clip(lower=0)
    out["transcricao_chars"] = out["call_id"].map(
        lambda c: len(str(ds.record(c).get(ds.transcript_col) or "")) if ds.transcript_col else 0)
    return out


# =============================================================================== estudo completo
@dataclass
class SingleCallStudy:
    out_dir: Path
    dataset: pd.DataFrame          # ligações + referência (gt) + baseline de produção
    calls: pd.DataFrame            # uma linha por ligação × abordagem (tokens, custo, latência, validade)
    preds: pd.DataFrame            # uma linha por ligação × KPI × abordagem
    judge: pd.DataFrame
    quality: pd.DataFrame          # por KPI: atual × novo × produção, diferença pareada com IC
    cost: pd.DataFrame             # por abordagem: tokens, custo medido/projetado, latência
    savings: dict                  # reduções do novo frente ao atual
    reference: str
    reference_mode: str
    reference_cost_usd: float
    model_single: str
    model_per_kpi: str
    notes: list[str] = field(default_factory=list)
    evals: dict[str, KPIEvaluation] = field(default_factory=dict)

    @property
    def json_path(self) -> Path:
        return self.out_dir / "saida_consolidada.jsonl"


def _frame_for_llm(cfg: BenchmarkConfig, calls: pd.DataFrame) -> pd.DataFrame:
    cols = ["call_id", "resumo", "transcricao"]
    # o provider simulado gera as respostas a partir dos rótulos do arquivo de exemplo
    extra = [k.gt_column for k in cfg.selected_kpis if cfg.run.mock and k.gt_column in calls.columns]
    return calls[cols + extra].copy()


def reference_labels(cfg: BenchmarkConfig, calls: pd.DataFrame, reference: str, mode: str, out_dir: Path,
                     *, kpis: list[KPISpec], max_tokens: int | None, max_palavras: int
                     ) -> tuple[pd.DataFrame, float]:
    """gt por ligação. Retorna (gt, custo_usd_da_referência)."""
    from kpibench.live import _dataset_from_frame

    if reference.startswith("arquivo:"):
        gt = pd.read_csv(reference.split(":", 1)[1], dtype={"call_id": str})
        return gt[["call_id"] + [k.gt_column for k in cfg.selected_kpis if k.gt_column in gt.columns]], 0.0
    ds = _dataset_from_frame(cfg, _frame_for_llm(cfg, calls), out_dir / "_entrada_referencia.csv")
    renderer = PromptRenderer(cfg.root / "prompts")
    gt = pd.DataFrame({"call_id": ds.ids()})
    if mode == "por_kpi":
        c = cfg.model_copy(deep=True)
        c.models = [reference]
        preds = run_kpi_extraction(c, ds, renderer, LLMExecutor(c, ds))
    elif mode == "consolidado":
        c = config_with_max_tokens(cfg, reference, max_tokens)
        _, preds = run_single_call(c, ds, reference, renderer, LLMExecutor(c, ds), kpis, label=reference,
                                   max_palavras=max_palavras)
    else:
        raise ValueError(f"modo de referência desconhecido: {mode} (use 'consolidado' ou 'por_kpi')")
    preds.to_csv(out_dir / "referencia_predicoes.csv", index=False)
    for k in cfg.selected_kpis:
        sub = preds[(preds["kpi"] == k.name) & preds["valid"]].set_index("call_id")["value"]
        gt[k.gt_column] = gt["call_id"].map(sub)
    return gt, float(preds["cost_usd"].sum())


def run_study(cfg: BenchmarkConfig, calls: pd.DataFrame, *, model_single: str = "gpt-6-luna",
              model_per_kpi: str = "gpt-6-luna", reference: str = "claude-opus-5-5",
              reference_mode: str = "consolidado", include_summary_quality: bool = True,
              max_tokens: int | None = 4000, max_palavras: int = 25, justificativas: bool = True,
              summary_cost_usd_per_call: float | None = None, kpi_weights: dict[str, float] | None = None,
              label: str = "extracao_consolidada") -> SingleCallStudy:
    """Roda referência, fluxo atual e fluxo consolidado nas mesmas ligações e compara qualidade e custo."""
    from kpibench.live import map_production_baseline
    from kpibench.pipeline import _baseline_text_rows

    calls = calls.copy()
    calls["call_id"] = calls["call_id"].astype(str)
    for c in ("resumo", "transcricao"):
        if c not in calls:
            calls[c] = ""
        calls[c] = calls[c].fillna("").astype(str)
    notes = []
    ok = (calls["resumo"].str.strip().str.len() > 0) & (calls["transcricao"].str.strip().str.len() > 0)
    if (~ok).any():
        notes.append(f"{int((~ok).sum())} ligações sem resumo ou sem transcrição foram excluídas "
                     "(os dois fluxos precisam ser medidos nas mesmas ligações).")
    calls = calls[ok].reset_index(drop=True)
    if calls.empty:
        raise ValueError("nenhuma ligação com resumo E transcrição")
    kpis_single = single_call_kpis(cfg, include_summary_quality)

    out_dir = cfg.resolve(cfg.run.output_dir) / f"{label}_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------- referência + dataset de avaliação
    gt, ref_cost = reference_labels(cfg, calls, reference, reference_mode, out_dir, kpis=kpis_single,
                                    max_tokens=max_tokens, max_palavras=max_palavras)
    if reference_mode == "consolidado" and not reference.startswith("arquivo:"):
        notes.append(f"Referência = {reference} com o prompt consolidado sobre a transcrição: vê toda a ligação, "
                     "mas compartilha o prompt do fluxo novo (tende a favorecê-lo). Para decisão final, use "
                     "rótulos auditados (REFERENCIA = 'arquivo:...').")
    elif reference_mode == "por_kpi" and not reference.startswith("arquivo:"):
        notes.append(f"Referência = {reference} com os prompts por KPI sobre o resumo: não vê o que o resumo "
                     "omitiu e compartilha o prompt do fluxo atual (tende a favorecê-lo).")
    prod, _ = map_production_baseline(calls)
    dataset = (_frame_for_llm(cfg, calls)[["call_id", "resumo", "transcricao"]]
               .merge(gt, on="call_id", how="left").merge(prod, on="call_id", how="left"))
    if "data_referencia" in calls:
        dataset = dataset.merge(calls[["call_id", "data_referencia"]], on="call_id", how="left")
    dataset.to_csv(out_dir / "dataset.csv", index=False)

    c_eval = cfg.model_copy(deep=True)
    c_eval.data.path = str(out_dir / "dataset.csv")
    c_eval.data.transcript_column = "transcricao"
    c_eval.data.reference_summary_column = None
    c_eval.data.baseline_prefix = "prod_"
    c_eval.data.sample = None
    ds = load_dataset(c_eval)
    notes += ds.warnings
    renderer = PromptRenderer(cfg.root / "prompts")

    # ------------------------------------------------------------- fluxo atual: 1 chamada por KPI (resumo)
    c_a = c_eval.model_copy(deep=True)
    c_a.models = [model_per_kpi]
    preds_a = run_kpi_extraction(c_a, ds, renderer, LLMExecutor(c_a, ds))
    calls_a = per_call_from_per_kpi(preds_a, ds)
    preds_a = preds_a.assign(modelo_llm=preds_a["model"], model=PER_KPI)

    # ------------------------------------------------------------- fluxo novo: 1 chamada por ligação
    c_b = config_with_max_tokens(c_eval, model_single, max_tokens)
    calls_b, preds_b = run_single_call(c_b, ds, model_single, renderer, LLMExecutor(c_b, ds), kpis_single,
                                       max_palavras=max_palavras, justificativas=justificativas)
    with open(out_dir / "saida_consolidada.jsonl", "w", encoding="utf-8") as fh:
        for s in calls_b["saida_json"]:
            fh.write(s + "\n")

    preds = pd.concat([preds_a, preds_b], ignore_index=True)
    all_calls = pd.concat([calls_a, calls_b.drop(columns="saida_json")], ignore_index=True)

    # ------------------------------------------------------------- juiz (KPIs de texto) + avaliação
    base_txt = _baseline_text_rows(c_eval, ds)
    judge_in = pd.concat([preds, base_txt], ignore_index=True) if not base_txt.empty else preds
    judge = run_reasoning_judge(c_eval, ds, judge_in, renderer, LLMExecutor(c_eval, ds))
    evals = {}
    for k in cfg.selected_kpis:
        ev = evaluate_kpi(c_eval, ds, k, preds, judge)
        if ev is not None:
            evals[k.name] = ev
    quality = quality_table(c_eval, evals)
    cost = cost_table(c_eval, all_calls, preds_a, model_per_kpi, model_single, summary_cost_usd_per_call)
    savings = savings_summary(cost, quality, kpi_weights)

    preds.to_csv(out_dir / "predicoes.csv", index=False)
    all_calls.to_csv(out_dir / "tokens_custo_por_ligacao.csv", index=False)
    judge.to_csv(out_dir / "juiz.csv", index=False)
    quality.to_csv(out_dir / "comparacao_qualidade.csv", index=False)
    cost.to_csv(out_dir / "comparacao_custo_tokens.csv", index=False)
    meta = {"modelo_consolidado": model_single, "modelo_por_kpi": model_per_kpi, "referencia": reference,
            "modo_referencia": reference_mode, "n_ligacoes": int(len(dataset)),
            "kpis_consolidado": [k.name for k in kpis_single], "max_tokens_consolidado": max_tokens,
            "custo_referencia_usd": ref_cost, "simulado": bool(cfg.run.mock), "reducoes": savings,
            "notas": notes, "criado_em": datetime.now().isoformat(timespec="seconds")}
    (out_dir / "estudo.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=float),
                                         encoding="utf-8")
    return SingleCallStudy(out_dir, dataset, all_calls, preds, judge, quality, cost, savings, reference,
                           reference_mode, ref_cost, model_single, model_per_kpi, notes, evals)


# =============================================================================== análises
def quality_table(cfg: BenchmarkConfig, evals: dict[str, KPIEvaluation]) -> pd.DataFrame:
    """Por KPI: métrica primária de cada abordagem (IC 95%) e diferença pareada novo − atual.

    decisão: melhora (IC da diferença > 0) · equivalente (IC dentro de ±margem) · nao_inferior
    (IC acima de −margem) · piora (IC abaixo de −margem) · inconclusivo (cruza −margem).
    """
    rows = []
    for name, ev in evals.items():
        t = ev.table.set_index("model")
        if PER_KPI not in t.index or SINGLE not in t.index:
            continue
        floor, ceiling = quality_bounds(cfg, ev)
        diff = ev.dists[SINGLE] - ev.dists[PER_KPI]
        lo, hi = percentile_ci(diff, cfg.statistics.ci)
        fin = diff[np.isfinite(diff)]
        m = ev.kpi.ni_margin
        if not np.isfinite(lo):
            dec = "sem_dados"
        elif lo > 0:
            dec = "melhora"
        elif hi < -m:
            dec = "piora"
        elif lo > -m and hi < m:
            dec = "equivalente"
        elif lo > -m:
            dec = "nao_inferior"
        else:
            dec = "inconclusivo"
        row = {"kpi": name, "display": ev.kpi.display, "tipo": ev.kpi.type, "metrica": ev.primary_name,
               "n": ev.n_items, "margem": m,
               "atual": t.at[PER_KPI, "primary"], "atual_ci_low": t.at[PER_KPI, "ci_low"],
               "atual_ci_high": t.at[PER_KPI, "ci_high"], "atual_validas": t.at[PER_KPI, "valid_rate"],
               "novo": t.at[SINGLE, "primary"], "novo_ci_low": t.at[SINGLE, "ci_low"],
               "novo_ci_high": t.at[SINGLE, "ci_high"], "novo_validas": t.at[SINGLE, "valid_rate"],
               "producao": t.at[BASELINE_NAME, "primary"] if BASELINE_NAME in t.index else np.nan,
               "diff_novo_menos_atual": t.at[SINGLE, "primary"] - t.at[PER_KPI, "primary"],
               "diff_ci_low": lo, "diff_ci_high": hi,
               "p_novo_melhor": float(np.mean(fin > 0)) if len(fin) else np.nan, "decisao": dec,
               "q_norm_atual": normalize_quality(t.at[PER_KPI, "primary"], floor, ceiling),
               "q_norm_novo": normalize_quality(t.at[SINGLE, "primary"], floor, ceiling)}
        rows.append(row)
    return pd.DataFrame(rows)


def _summ(g: pd.DataFrame, col: str) -> float:
    return float(g[col].mean()) if len(g) and col in g else np.nan


def cost_table(cfg: BenchmarkConfig, calls: pd.DataFrame, preds_per_kpi: pd.DataFrame, model_per_kpi: str,
               model_single: str, summary_cost_usd_per_call: float | None = None) -> pd.DataFrame:
    """Por abordagem: chamadas, tokens por ligação, custo medido e projetado (volume mensal, Batch, cache)."""
    pj = cfg.projection
    rows = []
    for ap, g in calls.groupby("abordagem", sort=False):
        ok = g[g["latency_s"].notna()]
        total_in = ok["input_tokens"] + ok["cached_tokens"] + ok["cache_write_tokens"]
        if ap == PER_KPI:
            spec = cfg.model(model_per_kpi)
            proj = 0.0
            for _, k in preds_per_kpi[preds_per_kpi["latency_s"].notna()].groupby("kpi"):
                tin = float((k["input_tokens"] + k["cached_tokens"] + k["cache_write_tokens"]).mean())
                proj += projected_cost_per_call(spec, tin, float(k["output_tokens"].mean()),
                                                float(k["static_prefix_tokens"].mean()), pj.cache_hit_ratio,
                                                pj.use_batch)
            cacheable = bool((preds_per_kpi.groupby("kpi")["static_prefix_tokens"].mean()
                              >= spec.min_cacheable_tokens).any())
        else:
            spec = cfg.model(model_single)
            prefix = _summ(ok, "prefixo_estatico_tokens")
            proj = projected_cost_per_call(spec, float(total_in.mean()) if len(ok) else np.nan,
                                           _summ(ok, "output_tokens"), prefix, pj.cache_hit_ratio, pj.use_batch)
            cacheable = bool(prefix >= spec.min_cacheable_tokens)
        measured = _summ(g, "cost_usd")
        row = {
            "abordagem": ap, "descricao": APPROACH_LABELS.get(ap, ap), "modelo_llm": spec.id, "ligacoes": len(g),
            "chamadas_por_ligacao": _summ(g, "chamadas_llm"),
            "entrada_tokens": float(total_in.mean()) if len(ok) else np.nan,
            "prefixo_estatico_tokens": _summ(ok, "prefixo_estatico_tokens"),
            "conteudo_variavel_tokens": _summ(ok, "conteudo_variavel_tokens"),
            "cache_lido_tokens": _summ(ok, "cached_tokens"),
            "saida_tokens": _summ(ok, "output_tokens"), "raciocinio_tokens": _summ(ok, "reasoning_tokens"),
            "total_tokens": float((total_in + ok["output_tokens"]).mean()) if len(ok) else np.nan,
            "prefixo_cacheavel": cacheable,
            "usd_por_ligacao_medido": measured, "usd_1k_medido": measured * 1000,
            "usd_1k_projetado": proj * 1000, "usd_mes_projetado": proj * pj.monthly_calls,
            "latencia_p50_s": float(ok["latencia_seq_s"].median()) if len(ok) else np.nan,
            "latencia_p95_s": float(ok["latencia_seq_s"].quantile(0.95)) if len(ok) else np.nan,
            "latencia_paralela_p50_s": float(ok["latencia_paralela_s"].median()) if len(ok) else np.nan,
            "kpis_validos_pct": float(g["kpis_validos"].sum() / g["n_kpis"].sum()) if g["n_kpis"].sum() else np.nan,
            "ligacoes_completas_pct": float((g["kpis_validos"] == g["n_kpis"]).mean()),
            "truncadas_pct": float(g["truncado"].fillna(False).astype(bool).mean()),
            "simulado": bool(g["simulated"].fillna(False).astype(bool).any()),
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    if summary_cost_usd_per_call is not None and (df["abordagem"] == PER_KPI).any():
        # o fluxo atual depende do resumo gerado antes (custo de outra chamada de LLM)
        extra = summary_cost_usd_per_call
        i = df.index[df["abordagem"] == PER_KPI][0]
        df["usd_1k_resumo_upstream"] = 0.0
        df.at[i, "usd_1k_resumo_upstream"] = extra * 1000
        for c in ("usd_1k_medido", "usd_1k_projetado"):
            df.at[i, c] += extra * 1000
        df.at[i, "usd_por_ligacao_medido"] += extra
        df.at[i, "usd_mes_projetado"] += extra * pj.monthly_calls
    return df


def _reduction(a: float, b: float) -> float:
    return float(100 * (1 - b / a)) if a and np.isfinite(a) and np.isfinite(b) else float("nan")


def savings_summary(cost: pd.DataFrame, quality: pd.DataFrame, kpi_weights: dict[str, float] | None = None
                    ) -> dict:
    """Reduções do fluxo novo frente ao atual (positivo = novo mais barato/menor) + qualidade agregada."""
    c = cost.set_index("abordagem")
    if PER_KPI not in c.index or SINGLE not in c.index:
        return {}
    a, b = c.loc[PER_KPI], c.loc[SINGLE]
    out = {
        "reducao_custo_medido_pct": _reduction(a["usd_1k_medido"], b["usd_1k_medido"]),
        "reducao_custo_projetado_pct": _reduction(a["usd_1k_projetado"], b["usd_1k_projetado"]),
        "reducao_tokens_totais_pct": _reduction(a["total_tokens"], b["total_tokens"]),
        "reducao_tokens_entrada_pct": _reduction(a["entrada_tokens"], b["entrada_tokens"]),
        "reducao_tokens_saida_pct": _reduction(a["saida_tokens"], b["saida_tokens"]),
        "reducao_chamadas_pct": _reduction(a["chamadas_por_ligacao"], b["chamadas_por_ligacao"]),
        "economia_mes_projetada_usd": float(a["usd_mes_projetado"] - b["usd_mes_projetado"]),
        "latencia_p50_atual_s": float(a["latencia_p50_s"]), "latencia_p50_novo_s": float(b["latencia_p50_s"]),
    }
    if not quality.empty:
        w = quality["kpi"].map(lambda k: (kpi_weights or {}).get(k, 1.0)).to_numpy(float)
        qa, qb = quality["q_norm_atual"].to_numpy(float), quality["q_norm_novo"].to_numpy(float)
        m = np.isfinite(qa) & np.isfinite(qb) & (w > 0)
        out.update({
            "qualidade_normalizada_atual": float(np.average(qa[m], weights=w[m])) if m.any() else np.nan,
            "qualidade_normalizada_novo": float(np.average(qb[m], weights=w[m])) if m.any() else np.nan,
            "kpis_melhora": int((quality["decisao"] == "melhora").sum()),
            "kpis_equivalentes_ou_nao_inferiores": int(quality["decisao"].isin(["equivalente", "nao_inferior"]).sum()),
            "kpis_piora": int((quality["decisao"] == "piora").sum()),
            "kpis_inconclusivos": int((quality["decisao"] == "inconclusivo").sum()),
        })
        out["ganho_qualidade_pp"] = 100 * (out["qualidade_normalizada_novo"] - out["qualidade_normalizada_atual"])
    return out


def tokens_by_kpi(preds: pd.DataFrame) -> pd.DataFrame:
    """Fluxo atual: tokens e custo médios por KPI (onde o custo se concentra)."""
    p = preds[preds["model"] == PER_KPI]
    if p.empty:
        return pd.DataFrame()
    g = p.groupby("kpi")
    out = pd.DataFrame({"entrada_tokens": g.apply(lambda d: (d["input_tokens"] + d["cached_tokens"]
                                                             + d["cache_write_tokens"]).mean(), include_groups=False),
                        "prefixo_estatico_tokens": g["static_prefix_tokens"].mean(),
                        "saida_tokens": g["output_tokens"].mean(), "raciocinio_tokens": g["reasoning_tokens"].mean(),
                        "usd_1k": g["cost_usd"].mean() * 1000})
    return out.reset_index().sort_values("usd_1k", ascending=False)


def executive_summary(study: SingleCallStudy) -> str:
    from kpibench.reporting.aa_charts import br, usd

    s, c = study.savings, study.cost.set_index("abordagem")
    if not s:
        return "_Sem resultados._"
    a, b = c.loc[PER_KPI], c.loc[SINGLE]
    L = [f"**{int(b['ligacoes'])} ligações · referência: `{study.reference}` ({study.reference_mode}) · "
         f"modelo: `{study.model_single}`**" + (" · **SIMULADO**" if bool(b["simulado"]) else ""), ""]
    L.append(f"- **Custo:** {usd(b['usd_1k_medido'])} por 1.000 ligações no fluxo novo contra "
             f"{usd(a['usd_1k_medido'])} no atual (**{br(s['reducao_custo_medido_pct'], 1)}%** de redução medida; "
             f"{br(s['reducao_custo_projetado_pct'], 1)}% na projeção de produção, com Batch/cache conforme "
             f"`projection`). Economia projetada: **{usd(s['economia_mes_projetada_usd'])}/mês**.")
    L.append(f"- **Tokens por ligação:** {br(b['total_tokens'], 0)} no novo contra {br(a['total_tokens'], 0)} no atual "
             f"({br(s['reducao_tokens_totais_pct'], 1)}% de redução). Entrada {br(b['entrada_tokens'], 0)} × "
             f"{br(a['entrada_tokens'], 0)}; saída {br(b['saida_tokens'], 0)} × {br(a['saida_tokens'], 0)}. "
             f"Chamadas: 1 × {br(a['chamadas_por_ligacao'], 0)}.")
    if "qualidade_normalizada_novo" in s:
        L.append(f"- **Qualidade (normalizada, 0% = acaso):** {br(100 * s['qualidade_normalizada_novo'], 1)}% no novo "
                 f"contra {br(100 * s['qualidade_normalizada_atual'], 1)}% no atual "
                 f"({'+' if s['ganho_qualidade_pp'] >= 0 else ''}{br(s['ganho_qualidade_pp'], 1)} pp). "
                 f"KPIs com melhora significativa: {s['kpis_melhora']}; equivalentes ou não-inferiores: "
                 f"{s['kpis_equivalentes_ou_nao_inferiores']}; piora: {s['kpis_piora']}; inconclusivos: "
                 f"{s['kpis_inconclusivos']}.")
    L.append(f"- **Confiabilidade do JSON consolidado:** {br(100 * b['ligacoes_completas_pct'], 1)}% das ligações com "
             f"todos os KPIs válidos; {br(100 * b['truncadas_pct'], 1)}% truncadas por `max_tokens`.")
    L.append(f"- **Latência por ligação (p50):** {br(b['latencia_p50_s'], 1)} s em 1 chamada, contra "
             f"{br(a['latencia_p50_s'], 1)} s somando as chamadas do fluxo atual "
             f"({br(a['latencia_paralela_p50_s'], 1)} s se disparadas em paralelo).")
    if bool(b["prefixo_cacheavel"]) and not bool(a["prefixo_cacheavel"]):
        L.append("- **Prompt caching:** o system prompt consolidado passa do mínimo cacheável e entra no cache "
                 "na projeção; os prompts por KPI, curtos, não.")
    for n in study.notes:
        L.append(f"- _{n}_")
    return "\n".join(L)


# =============================================================================== estimativa
def estimate_study_cost(cfg: BenchmarkConfig, calls: pd.DataFrame, *, model_single: str, model_per_kpi: str,
                        reference: str, reference_mode: str, include_summary_quality: bool = True,
                        max_palavras: int = 25) -> pd.DataFrame:
    """Custo estimado (antes de executar) a partir dos prompts reais destas ligações."""
    from kpibench.live import _dataset_from_frame

    frame = calls[["call_id", "resumo", "transcricao"]].fillna("")
    frame = frame[(frame["resumo"].str.len() > 0) & (frame["transcricao"].str.len() > 0)]
    ds = _dataset_from_frame(cfg, frame, cfg.root / ".cache" / "estimate_consolidado.csv")
    renderer = PromptRenderer(cfg.root / "prompts")
    kpis_single = single_call_kpis(cfg, include_summary_quality)
    n = len(ds.ids())
    in_per_kpi = sum(np.mean([estimate_tokens(p.system) + estimate_tokens(p.user) for p in
                              (renderer.render(f"kpis/{k.prompt}", **kpi_prompt_vars(k, ds, ds.record(c)))
                               for c in ds.ids())]) for k in cfg.selected_kpis) if n else 0.0
    out_per_kpi = sum(OUT_SUMMARY_QUALITY if k.name == "qualidade_resumo" else OUT_PER_KPI[k.type]
                      for k in cfg.selected_kpis)
    in_single = float(np.mean([estimate_tokens(p.system) + estimate_tokens(p.user) for p in
                               (render_single_call(renderer, kpis_single, r["transcricao"], r["resumo"], max_palavras)
                                for r in (ds.record(c) for c in ds.ids()))])) if n else 0.0
    out_single = 0.75 * sum(OUT_SUMMARY_QUALITY if k.name == "qualidade_resumo" else OUT_PER_KPI[k.type]
                            for k in kpis_single)

    def usd(model, tin, tout, n_calls_per_lig):
        spec = cfg.model(model)
        reasoning = REASONING_OVERHEAD * n_calls_per_lig if "reasoning_effort" in spec.params else 0
        return n * (tin * spec.pricing.input + (tout + reasoning) * spec.pricing.output) / 1e6

    k = len(cfg.selected_kpis)
    rows = [{"papel": "fluxo atual (1 chamada por KPI, resumo)", "modelo": model_per_kpi, "chamadas_llm": n * k,
             "entrada_tokens_por_ligacao": in_per_kpi, "saida_tokens_por_ligacao": out_per_kpi,
             "usd_estimado": usd(model_per_kpi, in_per_kpi, out_per_kpi, k)},
            {"papel": "fluxo novo (1 chamada por ligação, transcrição)", "modelo": model_single, "chamadas_llm": n,
             "entrada_tokens_por_ligacao": in_single, "saida_tokens_por_ligacao": out_single,
             "usd_estimado": usd(model_single, in_single, out_single, 1)}]
    if not reference.startswith("arquivo:"):
        if reference_mode == "consolidado":
            rows.append({"papel": "referência (consolidado)", "modelo": reference, "chamadas_llm": n,
                         "entrada_tokens_por_ligacao": in_single, "saida_tokens_por_ligacao": out_single,
                         "usd_estimado": usd(reference, in_single, out_single, 1)})
        else:
            rows.append({"papel": "referência (por KPI)", "modelo": reference, "chamadas_llm": n * k,
                         "entrada_tokens_por_ligacao": in_per_kpi, "saida_tokens_por_ligacao": out_per_kpi,
                         "usd_estimado": usd(reference, in_per_kpi, out_per_kpi, k)})
    text_kpis = [x for x in cfg.selected_kpis if x.type == "text" and x.judge_prompt]
    if cfg.judge.enabled and cfg.judge.model and text_kpis:
        spec = cfg.model(cfg.judge.model)
        nj = n * len(text_kpis) * 3
        rows.append({"papel": "juiz (KPIs de texto)", "modelo": cfg.judge.model, "chamadas_llm": nj,
                     "entrada_tokens_por_ligacao": 700 * len(text_kpis) * 3, "saida_tokens_por_ligacao": 120 * len(text_kpis) * 3,
                     "usd_estimado": nj * (700 * spec.pricing.input + 120 * spec.pricing.output) / 1e6})
    df = pd.DataFrame(rows)
    df.loc[len(df)] = {"papel": "TOTAL", "modelo": "", "chamadas_llm": int(df["chamadas_llm"].sum()),
                       "entrada_tokens_por_ligacao": np.nan, "saida_tokens_por_ligacao": np.nan,
                       "usd_estimado": float(df["usd_estimado"].sum())}
    return df


# =============================================================================== relatório final
COST_COMPONENTS = [("input_tokens", "entrada (preço cheio)", "input"),
                   ("cached_tokens", "entrada lida do cache", "cached_input"),
                   ("cache_write_tokens", "escrita de cache", "cache_write"),
                   ("saida_visivel_tokens", "saída (JSON)", "output"),
                   ("reasoning_tokens", "raciocínio", "output")]


def cost_breakdown(cfg: BenchmarkConfig, study: SingleCallStudy) -> pd.DataFrame:
    """Cálculo do custo medido por componente: tokens médios por ligação × preço por 1M tokens.

    A soma dos componentes de cada abordagem reproduz o custo medido (mesma fórmula de
    `llm.pricing.cost_usd`); raciocínio é cobrado como saída.
    """
    rows = []
    for ap, model in ((PER_KPI, study.model_per_kpi), (SINGLE, study.model_single)):
        g = study.calls[(study.calls["abordagem"] == ap) & study.calls["latency_s"].notna()]
        if g.empty:
            continue
        p = cfg.model(model).pricing
        price = {"input": p.input, "cached_input": p.cached_input,
                 "cache_write": p.input * p.cache_write_multiplier, "output": p.output}
        tok = {c: float(g[c].mean()) for c in ("input_tokens", "cached_tokens", "cache_write_tokens", "reasoning_tokens")}
        tok["saida_visivel_tokens"] = float((g["output_tokens"] - g["reasoning_tokens"]).clip(lower=0).mean())
        for col, label, key in COST_COMPONENTS:
            usd_call = tok[col] * price[key] / 1e6
            rows.append({"abordagem": ap, "descricao": APPROACH_LABELS[ap], "modelo_llm": model, "componente": label,
                         "tokens_por_ligacao": tok[col], "preco_usd_por_1M": price[key],
                         "usd_por_ligacao": usd_call, "usd_1k_ligacoes": usd_call * 1000})
    df = pd.DataFrame(rows)
    if not df.empty:
        tot = df.groupby("abordagem")["usd_1k_ligacoes"].transform("sum")
        df["pct_do_custo"] = np.where(tot > 0, 100 * df["usd_1k_ligacoes"] / tot, np.nan)
    return df


def aggregate_quality(cfg: BenchmarkConfig, study: SingleCallStudy, weights: dict[str, float] | None = None
                      ) -> pd.DataFrame:
    """Qualidade normalizada média (0 = acaso, 1 = teto) por abordagem, com IC bootstrap pareado, e a
    diferença novo − atual (as mesmas réplicas valem para todos os KPIs e abordagens)."""
    reps, ws = {PER_KPI: [], SINGLE: []}, []
    for name, ev in study.evals.items():
        if PER_KPI not in ev.dists or SINGLE not in ev.dists:
            continue
        floor, ceiling = quality_bounds(cfg, ev)
        for ap in reps:
            reps[ap].append(normalize_quality(ev.dists[ap], floor, ceiling))
        ws.append((weights or {}).get(name, 1.0))
    if not ws or study.quality.empty:
        return pd.DataFrame()
    w = np.array(ws, float)

    def agg(stack):
        return np.ma.average(np.ma.masked_invalid(np.vstack(stack)), axis=0, weights=w).filled(np.nan)

    a, b = agg(reps[PER_KPI]), agg(reps[SINGLE])
    q = study.quality
    wq = q["kpi"].map(lambda k: (weights or {}).get(k, 1.0)).to_numpy(float)
    point = {PER_KPI: float(np.average(q["q_norm_atual"], weights=wq)),
             SINGLE: float(np.average(q["q_norm_novo"], weights=wq))}
    rows = []
    for ap, d in ((PER_KPI, a), (SINGLE, b), ("diferenca", b - a)):
        lo, hi = percentile_ci(d, cfg.statistics.ci)
        val = point[SINGLE] - point[PER_KPI] if ap == "diferenca" else point[ap]
        fin = d[np.isfinite(d)]
        rows.append({"abordagem": ap, "q_norm": val, "ci_low": lo, "ci_high": hi,
                     "p_maior_que_zero": float(np.mean(fin > 0)) if ap == "diferenca" and len(fin) else np.nan})
    return pd.DataFrame(rows)


def per_kpi_cost_quality(study: SingleCallStudy) -> pd.DataFrame:
    """Por KPI: custo por 1.000 ligações (atual = a chamada do KPI; novo = rateio igual da chamada única)
    e qualidade normalizada de cada abordagem."""
    p = study.preds
    cost = (p[p["model"].isin([PER_KPI, SINGLE])].groupby(["kpi", "model"])["cost_usd"].mean() * 1000).unstack()
    q = study.quality.set_index("kpi")
    out = pd.DataFrame({"display": q["display"], "usd_1k_atual": cost.get(PER_KPI), "usd_1k_novo": cost.get(SINGLE),
                        "q_atual": q["q_norm_atual"], "q_novo": q["q_norm_novo"], "decisao": q["decisao"]})
    return out.reset_index()


def quality_cost_summary(cfg: BenchmarkConfig, study: SingleCallStudy, weights: dict[str, float] | None = None
                         ) -> dict:
    """Relação qualidade × custo das duas abordagens e leitura da troca (dominância ou ICER)."""
    c = study.cost.set_index("abordagem")
    aq = aggregate_quality(cfg, study, weights)
    if aq.empty or PER_KPI not in c.index or SINGLE not in c.index:
        return {}
    aq = aq.set_index("abordagem")
    qa, qb = aq.at[PER_KPI, "q_norm"], aq.at[SINGLE, "q_norm"]
    ca, cb = c.at[PER_KPI, "usd_1k_medido"], c.at[SINGLE, "usd_1k_medido"]
    dq, dc = 100 * (qb - qa), cb - ca                      # pp de qualidade; US$ por 1k ligações
    lo, hi = aq.at["diferenca", "ci_low"], aq.at["diferenca", "ci_high"]
    from kpibench.reporting.aa_charts import br

    def m(v):
        return f"US$ {br(v, 4 if abs(v) < 1 else 2)}"

    if dc <= 0 and dq >= 0:
        verdict, text = "domina", (f"o fluxo novo **domina** o atual: custa {m(abs(dc))} a menos por 1.000 "
                                   f"ligações e entrega {'+' if dq >= 0 else '−'}{br(abs(dq), 1)} pp de qualidade")
    elif dc < 0 and dq < 0:
        verdict, text = "troca_economia", (f"o fluxo novo **economiza {m(abs(dc))} por 1.000 ligações** e perde "
                                           f"{br(abs(dq), 1)} pp de qualidade: {m(abs(dc) / abs(dq))} economizados "
                                           "por 1.000 ligações para cada pp perdido")
    elif dc > 0 and dq > 0:
        verdict, text = "troca_qualidade", (f"o fluxo novo **custa {m(dc)} a mais por 1.000 ligações** e ganha "
                                            f"{br(dq, 1)} pp: ICER de {m(dc / dq)} por 1.000 ligações por pp")
    else:
        verdict, text = "dominado", "o fluxo novo é **dominado**: custa mais e não melhora a qualidade"
    return {"q_atual": qa, "q_novo": qb, "q_atual_ci": (aq.at[PER_KPI, "ci_low"], aq.at[PER_KPI, "ci_high"]),
            "q_novo_ci": (aq.at[SINGLE, "ci_low"], aq.at[SINGLE, "ci_high"]),
            "diff_pp": dq, "diff_ci_pp": (100 * lo, 100 * hi), "p_novo_melhor": aq.at["diferenca", "p_maior_que_zero"],
            "diff_significativa": bool(lo > 0 or hi < 0),
            "usd_1k_atual": ca, "usd_1k_novo": cb, "diff_usd_1k": dc,
            "usd_1k_por_pp_atual": ca / (100 * qa) if qa > 0 else np.nan,
            "usd_1k_por_pp_novo": cb / (100 * qb) if qb > 0 else np.nan,
            "pp_por_usd_1k_atual": 100 * qa / ca if ca > 0 else np.nan,
            "pp_por_usd_1k_novo": 100 * qb / cb if cb > 0 else np.nan,
            "veredito": verdict, "leitura": text}


def final_report(cfg: BenchmarkConfig, study: SingleCallStudy, weights: dict[str, float] | None = None) -> str:
    """Relatório em markdown: uma LLM por KPI × todos os KPIs numa chamada (custo, cálculo, tokens, qualidade)."""
    from kpibench.reporting.aa_charts import br, usd

    c = study.cost.set_index("abordagem")
    if PER_KPI not in c.index or SINGLE not in c.index:
        return "_Sem as duas abordagens para comparar._"
    a, b, s = c.loc[PER_KPI], c.loc[SINGLE], study.savings
    qc = quality_cost_summary(cfg, study, weights)
    bd = cost_breakdown(cfg, study)
    pj = cfg.projection
    n_kpis = int(round(a["chamadas_por_ligacao"]))
    year = 12 * pj.monthly_calls
    pct = lambda v: f"{br(v, 1)}%"  # noqa: E731
    fine = lambda v: f"US$ {br(v, 6)}"  # noqa: E731  (valores por ligação são frações de centavo)

    def signed(f):
        return lambda v: ("−" if v < 0 else "+") + f(abs(v)).replace("US$ -", "US$ ")

    def md_table(df):
        return df.to_markdown(index=False, disable_numparse=True)

    L = ["# Relatório: uma LLM por KPI × todos os KPIs numa chamada", ""]
    if bool(b["simulado"]):
        L += ["> ⚠️ **RODADA SIMULADA.** Saídas geradas pelo provider mock; tokens vêm dos prompts reais. "
              "Use só para validar o fluxo, não para decidir.", ""]
    L += [f"**{int(b['ligacoes'])} ligações** · atual: `{study.model_per_kpi}`, {n_kpis} chamadas por ligação sobre "
          f"o resumo · novo: `{study.model_single}`, 1 chamada por ligação sobre a transcrição completa · "
          f"referência: `{study.reference}` ({study.reference_mode}).", ""]

    # ---------------------------------------------------------------- 1. diferença de custo
    L += ["## 1. Diferença de custo", ""]
    rows = [("US$ por ligação (medido)", a["usd_por_ligacao_medido"], b["usd_por_ligacao_medido"], fine, -s["reducao_custo_medido_pct"]),
            ("US$ por 1.000 ligações (medido)", a["usd_1k_medido"], b["usd_1k_medido"], usd, -s["reducao_custo_medido_pct"]),
            ("US$ por 1.000 ligações (projetado)", a["usd_1k_projetado"], b["usd_1k_projetado"], usd, -s["reducao_custo_projetado_pct"]),
            (f"US$ por mês ({br(pj.monthly_calls, 0)} ligações, projetado)", a["usd_mes_projetado"], b["usd_mes_projetado"], usd,
             -s["reducao_custo_projetado_pct"]),
            ("US$ por ano (projetado)", a["usd_1k_projetado"] * year / 1000, b["usd_1k_projetado"] * year / 1000, usd,
             -s["reducao_custo_projetado_pct"]),
            ("chamadas de LLM por ligação", a["chamadas_por_ligacao"], b["chamadas_por_ligacao"], lambda v: br(v, 0),
             -s["reducao_chamadas_pct"]),
            ("tokens por ligação", a["total_tokens"], b["total_tokens"], lambda v: br(v, 0), -s["reducao_tokens_totais_pct"])]
    tab = pd.DataFrame([{"": lab, "Uma LLM por KPI": f(x), "Todos os KPIs consolidados": f(y), "Diferença (novo − atual)": signed(f)(y - x),
                         "Variação": pct(v)} for lab, x, y, f, v in rows])
    L += [md_table(tab), ""]
    verb = "reduz" if s["reducao_custo_medido_pct"] > 0 else "aumenta"
    L += [f"**Leitura:** consolidar os KPIs numa chamada {verb} o custo medido em "
          f"**{pct(abs(s['reducao_custo_medido_pct']))}** ({usd(a['usd_1k_medido'])} → {usd(b['usd_1k_medido'])} por "
          f"1.000 ligações). Na projeção de produção ({'Batch' if pj.use_batch else 'API síncrona'}, cache hit "
          f"{pj.cache_hit_ratio:.0%} no prefixo cacheável) a diferença é de **{pct(s['reducao_custo_projetado_pct'])}**: "
          f"**{usd(s['economia_mes_projetada_usd'])} por mês** e {usd(12 * s['economia_mes_projetada_usd'])} por ano.", ""]
    if "usd_1k_resumo_upstream" in c.columns:
        L += [f"O custo do fluxo atual inclui {usd(a['usd_1k_resumo_upstream'])}/1k para gerar o resumo, do qual ele "
              "depende.", ""]

    # ---------------------------------------------------------------- 2. cálculo
    L += ["## 2. Cálculo do custo e consumo de tokens", "",
          "Custo por ligação = Σ (tokens médios do componente × preço do componente por 1M tokens) ÷ 1.000.000. "
          "Raciocínio é cobrado como saída. Tokens medidos pelo `usage` das chamadas; preços de `config/models.yaml`.",
          ""]
    for ap in (PER_KPI, SINGLE):
        d = bd[(bd["abordagem"] == ap) & (bd["tokens_por_ligacao"] > 0)]
        if d.empty:
            continue
        terms = " + ".join(f"{br(r.tokens_por_ligacao, 0)} × US$ {br(r.preco_usd_por_1M, 3)}" for r in d.itertuples())
        total = float(d["usd_por_ligacao"].sum())
        L += [f"- **{APPROACH_LABELS[ap]}** (`{d['modelo_llm'].iloc[0]}`): ({terms}) ÷ 1.000.000 = "
              f"**{fine(total)} por ligação** = {usd(1000 * total)} por 1.000 ligações."]
    L.append("")
    view = pd.DataFrame({"abordagem": bd["descricao"], "componente": bd["componente"],
                         "tokens/ligação": bd["tokens_por_ligacao"].map(lambda v: br(v, 0)),
                         "US$ por 1M": bd["preco_usd_por_1M"].map(lambda v: br(v, 3)),
                         "US$/1k ligações": bd["usd_1k_ligacoes"].map(usd),
                         "% do custo": bd["pct_do_custo"].map(pct)})
    L += [md_table(view), ""]
    L += [f"Onde estão os tokens (por ligação, atual × novo): instruções {br(a['prefixo_estatico_tokens'], 0)} × "
          f"{br(b['prefixo_estatico_tokens'], 0)} — o fluxo atual reenvia instruções em cada uma das {n_kpis} chamadas; "
          f"conteúdo {br(a['conteudo_variavel_tokens'], 0)} × {br(b['conteudo_variavel_tokens'], 0)} — resumo "
          f"repetido {n_kpis} vezes × transcrição lida uma vez; saída {br(a['saida_tokens'], 0)} × "
          f"{br(b['saida_tokens'], 0)}.", ""]
    if bool(b["prefixo_cacheavel"]):
        L += [f"O system prompt consolidado ({br(b['prefixo_estatico_tokens'], 0)} tokens) passa do mínimo cacheável: em "
              f"produção, {pj.cache_hit_ratio:.0%} dele sai a preço de cache — parte da diferença entre medido e "
              "projetado.", ""]

    # ---------------------------------------------------------------- 3. qualidade × custo
    L += ["## 3. Relação qualidade × custo", ""]
    if qc:
        ci = lambda t: f"[{br(100 * t[0], 1)}; {br(100 * t[1], 1)}]"  # noqa: E731
        tab = pd.DataFrame({
            "": ["qualidade normalizada média (IC 95%)", "US$ por 1.000 ligações (medido)",
                 "US$/1k por ponto de qualidade", "pontos de qualidade por US$ 1/1k"],
            "Uma LLM por KPI": [f"{pct(100 * qc['q_atual'])} {ci(qc['q_atual_ci'])}", usd(qc["usd_1k_atual"]),
                                br(qc["usd_1k_por_pp_atual"], 4), br(qc["pp_por_usd_1k_atual"], 1)],
            "Todos os KPIs consolidados": [f"{pct(100 * qc['q_novo'])} {ci(qc['q_novo_ci'])}", usd(qc["usd_1k_novo"]),
                                           br(qc["usd_1k_por_pp_novo"], 4), br(qc["pp_por_usd_1k_novo"], 1)],
        })
        L += [md_table(tab), ""]
        sig = ("diferença estatisticamente significativa" if qc["diff_significativa"]
               else "diferença **não** significativa com esta amostra")
        L += ["Qualidade normalizada: 0% = acaso, 100% = concordância total com a referência; média dos KPIs. "
              f"Diferença novo − atual: **{'+' if qc['diff_pp'] >= 0 else '−'}{br(abs(qc['diff_pp']), 1)} pp** (IC 95% {br(qc['diff_ci_pp'][0], 1)} a "
              f"{br(qc['diff_ci_pp'][1], 1)} pp; P(novo melhor) = {br(100 * qc['p_novo_melhor'], 0)}%) — {sig}.", "",
              f"**Veredito:** {qc['leitura']}.", ""]
        q = study.quality
        L += [f"Por KPI: {int((q['decisao'] == 'melhora').sum())} com melhora significativa, "
              f"{int(q['decisao'].isin(['equivalente', 'nao_inferior']).sum())} equivalentes ou não-inferiores, "
              f"{int((q['decisao'] == 'piora').sum())} com piora e {int((q['decisao'] == 'inconclusivo').sum())} "
              "inconclusivos (detalhe na seção 10).", ""]
    else:
        L += ["_Sem dados de qualidade para as duas abordagens._", ""]

    # ---------------------------------------------------------------- 4. operação e ressalvas
    L += ["## 4. Operação e ressalvas", "",
          f"- **Confiabilidade:** {pct(100 * b['ligacoes_completas_pct'])} das ligações com todos os KPIs válidos no "
          f"JSON consolidado ({pct(100 * a['ligacoes_completas_pct'])} no fluxo atual); "
          f"{pct(100 * b['truncadas_pct'])} truncadas por `max_tokens`. No fluxo novo, uma falha de chamada perde "
          "todos os KPIs da ligação; no atual, só um.",
          f"- **Latência (p50 por ligação):** {br(b['latencia_p50_s'], 1)} s × {br(a['latencia_p50_s'], 1)} s em "
          f"sequência ({br(a['latencia_paralela_p50_s'], 1)} s em paralelo).",
          "- **Tamanho da transcrição:** o custo do fluxo novo cresce com a transcrição (seção 9.3); confira o p90 "
          "das ligações de produção antes de projetar.",
          "- **Custo do resumo:** o fluxo atual depende de um resumo gerado antes"
          + ("; esse custo está incluído." if "usd_1k_resumo_upstream" in c.columns else
             ", não incluído aqui (`CUSTO_RESUMO_USD_POR_LIGACAO`); com ele, a economia do fluxo novo é maior.")]
    L += [f"- {n}" for n in study.notes]
    return "\n".join(L)
