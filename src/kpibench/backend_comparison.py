"""Qual backend compensa mais na extração de KPIs: LLM (atual, por KPI) × LLM consolidado × CDM × Jev ×
híbrido. Mesmas ligações, mesma referência; mede latência, custo e a relação qualidade × custo (notebook 05).

Qualidade: métrica primária por KPI contra a referência (`evaluate_kpi`), normalizada do acaso ao teto e
agregada nos KPIs que TODOS os backends cobrem (o Jev não gera texto livre, então a comparação justa é
nos KPIs estruturados); a cobertura de cada backend é reportada à parte. Diferenças contra a base
(sistema atual) por bootstrap pareado. Custo: medido (usage × preço) e projetado em produção por
requisição (Batch e cache do prefixo conforme `projection`). Latência: por ligação, como em produção.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from kpibench import extractors as X
from kpibench import single_call as sc
from kpibench.analysis.cost_effectiveness import frontier_icer, normalize_quality, quality_bounds
from kpibench.config import BenchmarkConfig
from kpibench.extraction.runner import LLMExecutor, run_reasoning_judge
from kpibench.llm.pricing import projected_cost_per_call
from kpibench.metrics.kpi_eval import BASELINE_NAME, KPIEvaluation, evaluate_kpi
from kpibench.metrics.stats import percentile_ci

KIND_LABELS = {"llm_per_kpi": "LLM, 1 chamada por KPI (atual)", "llm_single_call": "LLM consolidado",
               "llm_json_schema": "LLM com structured outputs", "cdm": "CDM ScaleDown (Decisions + Summarize)",
               "jev": "Jev (TypeSafe)", "hybrid_cdm": "Híbrido CDM ScaleDown + LLM", "hybrid_jev": "Híbrido Jev + LLM"}
COST_COMPONENTS = [("input_tokens", "entrada (preço cheio)", "input"),
                   ("cached_tokens", "entrada lida do cache", "cached_input"),
                   ("cache_write_tokens", "escrita de cache", "cache_write"),
                   ("saida_visivel_tokens", "saída", "output"),
                   ("reasoning_tokens", "raciocínio", "output")]


@dataclass
class BackendStudy:
    out_dir: Path
    cfg: BenchmarkConfig                 # cfg de avaliação, com os ModelSpecs de CDM/Jev registrados
    dataset: pd.DataFrame
    backends: list[str]
    baseline: str
    kinds: dict[str, str]
    coverage: dict[str, list[str]]       # KPIs que cada backend responde
    common_kpis: list[str]               # KPIs cobertos por todos (base da comparação de qualidade)
    calls: pd.DataFrame                  # backend × ligação: requisições, tokens, custo, latência
    preds: pd.DataFrame                  # backend × ligação × KPI
    requests: pd.DataFrame               # uma linha por requisição de API
    judge: pd.DataFrame
    quality: pd.DataFrame                # backend × KPI: métrica primária, IC, q normalizada
    comparisons: pd.DataFrame            # backend × KPI: diferença pareada contra a base
    summary: pd.DataFrame                # uma linha por backend: custo, latência, qualidade, fronteira
    breakdown: pd.DataFrame              # cálculo do custo: backend × modelo × componente
    calibration: pd.DataFrame            # confiança × acerto (backends que informam confiança)
    recommendation: dict
    reference: str
    reference_mode: str
    reference_cost_usd: float
    quality_margin_pp: float
    sla_s: float | None
    notes: list[str] = field(default_factory=list)
    evals: dict[str, KPIEvaluation] = field(default_factory=dict)

    @property
    def simulated(self) -> bool:
        return bool(self.calls["simulated"].any()) if "simulated" in self.calls else False

    def jsonl_path(self, backend: str) -> Path:
        return self.out_dir / f"saida_{backend}.jsonl"


# =============================================================================== execução
def estimate_backends(cfg: BenchmarkConfig, calls: pd.DataFrame, backends: list[str],
                      include_summary_quality: bool = True) -> pd.DataFrame:
    """Custo e tokens previstos por backend para estas ligações (sem chamar APIs)."""
    calls, _ = sc.clean_calls(calls, need_summary=False)
    ctx = X.ExtractorContext(cfg)
    kpis = sc.single_call_kpis(cfg, include_summary_quality)
    records = calls.to_dict("records")
    rows = [X.estimate_cost(X.build_extractor(b, ctx, kpis), records) for b in backends]
    df = pd.DataFrame(rows)
    df["preco_estimado"] = [any(ctx.cfg.model(m).pricing_is_estimate for m in _models_of(ctx, b)) for b in backends]
    return df


def _models_of(ctx: X.ExtractorContext, backend: str) -> list[str]:
    ex = ctx.built[backend]
    if isinstance(ex, X.HybridExtractor):
        return _models_of(ctx, ex.primary.name) + _models_of(ctx, ex.secondary.name)
    return [ex.spec.id] + ([ex.summ_spec.id] if isinstance(ex, X.ScaleDownCDMExtractor) else [])


def run_backend_comparison(cfg: BenchmarkConfig, calls: pd.DataFrame, backends: list[str], *,
                           baseline: str = "llm_por_kpi", reference: str = "claude-opus-5-5",
                           reference_mode: str = "consolidado", include_summary_quality: bool = True,
                           weights: dict[str, float] | None = None, quality_margin_pp: float = 3.0,
                           sla_s: float | None = None, workers: int | None = None,
                           label: str = "comparacao_backends") -> BackendStudy:
    if baseline not in backends:
        backends = [baseline] + list(backends)
    conf = X.load_extractor_config(cfg)
    need_summary = include_summary_quality or any(conf[b]["type"] == "llm_per_kpi" for b in backends)
    calls, notes = sc.clean_calls(calls, need_summary=need_summary)
    kpis = sc.single_call_kpis(cfg, include_summary_quality)
    out_dir = cfg.resolve(cfg.run.output_dir) / f"{label}_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------- referência e dataset de avaliação
    gt, ref_cost = sc.reference_labels(cfg, calls, reference, reference_mode, out_dir, kpis=kpis,
                                       max_tokens=4000, max_palavras=25)
    notes += sc.reference_notes(reference, reference_mode)
    dataset, ds, c_eval = sc.build_eval_dataset(cfg, calls, gt, out_dir)
    c_eval.kpis = [k.name for k in kpis]
    notes += ds.warnings

    # ------------------------------------------------------------- backends
    ctx = X.ExtractorContext(c_eval, ds, conf)
    records = [ds.record(c) for c in ds.ids()]
    all_exts, kinds, coverage = [], {}, {}
    for b in backends:
        ex = X.build_extractor(b, ctx, kpis)
        exts = X.run_extractor(ex, records, workers or cfg.run.max_workers)
        X.write_jsonl(exts, out_dir / f"saida_{b}.jsonl")
        all_exts += exts
        kinds[b] = ex.kind
        coverage[b] = [k.name for k in kpis if ex.supports(k)]
    per_call, preds, reqs = X.extraction_frames(all_exts)

    # ------------------------------------------------------------- juiz e avaliação
    from kpibench.pipeline import _baseline_text_rows

    renderer = ctx.renderer
    base_txt = _baseline_text_rows(c_eval, ds)
    judge_in = pd.concat([preds, base_txt], ignore_index=True) if not base_txt.empty else preds
    judge = run_reasoning_judge(c_eval, ds, judge_in, renderer, LLMExecutor(c_eval, ds))
    evals = {}
    for k in kpis:
        ev = evaluate_kpi(c_eval, ds, k, preds, judge)
        if ev is not None:
            evals[k.name] = ev
    common = [k for k in evals if all(k in coverage[b] for b in backends)]

    quality = quality_table(c_eval, evals, backends)
    comps = comparisons_table(c_eval, evals, backends, baseline)
    summary = summary_table(ctx, per_call, reqs, backends, kinds, coverage, len(kpis), sla_s)
    agg = aggregate_quality(c_eval, evals, backends, baseline, common, weights)
    summary = summary.merge(agg, on="backend", how="left")
    summary["decisao_vs_base"] = [sc.paired_decision(lo / 100, hi / 100, quality_margin_pp / 100)
                                  if b != baseline else "base" for b, lo, hi in
                                  zip(summary["backend"], summary["diff_base_ci_low_pp"], summary["diff_base_ci_high_pp"])]
    summary["usd_1k_por_pp"] = summary["usd_1k_projetado"] / (100 * summary["q_comum"]).where(summary["q_comum"] > 0)
    fr = frontier_icer(summary[["backend", "usd_1k_projetado", "q_comum"]], cost_col="usd_1k_projetado",
                       q_col="q_comum", model_col="backend")
    summary = summary.merge(fr[["backend", "status_ce", "icer_usd_1k_por_pp", "vizinho_anterior"]], on="backend")
    breakdown = cost_breakdown(ctx, reqs, per_call)
    calib = calibration_table(c_eval, preds, ds, kpis)
    rec = recommend(summary, baseline, len(kpis))

    study = BackendStudy(out_dir, ctx.cfg, dataset, backends, baseline, kinds, coverage, common, per_call, preds,
                         reqs, judge, quality, comps, summary, breakdown, calib, rec, reference, reference_mode,
                         ref_cost, quality_margin_pp, sla_s, notes, evals)
    _save(study)
    return study


def _save(st: BackendStudy) -> None:
    o = st.out_dir
    st.calls.to_csv(o / "por_ligacao.csv", index=False)
    st.preds.to_csv(o / "predicoes.csv", index=False)
    st.requests.to_csv(o / "requisicoes.csv", index=False)
    st.judge.to_csv(o / "juiz.csv", index=False)
    st.quality.to_csv(o / "qualidade_por_kpi.csv", index=False)
    st.comparisons.to_csv(o / "comparacao_vs_base.csv", index=False)
    st.summary.to_csv(o / "resumo_backends.csv", index=False)
    st.breakdown.to_csv(o / "calculo_custo_componentes.csv", index=False)
    st.calibration.to_csv(o / "calibracao_confianca.csv", index=False)
    meta = {"backends": st.backends, "base": st.baseline, "tipos": st.kinds, "cobertura": st.coverage,
            "kpis_comuns": st.common_kpis, "referencia": st.reference, "modo_referencia": st.reference_mode,
            "custo_referencia_usd": st.reference_cost_usd, "n_ligacoes": int(len(st.dataset)),
            "margem_qualidade_pp": st.quality_margin_pp, "sla_s": st.sla_s, "simulado": st.simulated,
            "recomendacao": st.recommendation, "notas": st.notes, "criado_em": datetime.now().isoformat(timespec="seconds")}
    (o / "estudo.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


# =============================================================================== qualidade
def quality_table(cfg: BenchmarkConfig, evals: dict[str, KPIEvaluation], backends: list[str]) -> pd.DataFrame:
    rows = []
    for name, ev in evals.items():
        floor, ceiling = quality_bounds(cfg, ev)
        t = ev.table.set_index("model")
        for b in backends + [BASELINE_NAME]:
            if b not in t.index:
                continue
            r = t.loc[b]
            rows.append({"kpi": name, "display": ev.kpi.display, "tipo": ev.kpi.type, "metrica": ev.primary_name,
                         "backend": b, "primary": r["primary"], "ci_low": r["ci_low"], "ci_high": r["ci_high"],
                         "valid_rate": r["valid_rate"], "q_norm": normalize_quality(r["primary"], floor, ceiling),
                         "q_ci_low": normalize_quality(r["ci_low"], floor, ceiling),
                         "q_ci_high": normalize_quality(r["ci_high"], floor, ceiling)})
    return pd.DataFrame(rows)


def comparisons_table(cfg: BenchmarkConfig, evals: dict[str, KPIEvaluation], backends: list[str],
                      baseline: str) -> pd.DataFrame:
    rows = []
    for name, ev in evals.items():
        if baseline not in ev.dists:
            continue
        t = ev.table.set_index("model")
        for b in backends:
            if b == baseline or b not in ev.dists:
                continue
            d = ev.dists[b] - ev.dists[baseline]
            lo, hi = percentile_ci(d, cfg.statistics.ci)
            fin = d[np.isfinite(d)]
            rows.append({"kpi": name, "display": ev.kpi.display, "metrica": ev.primary_name, "backend": b,
                         "diff": t.at[b, "primary"] - t.at[baseline, "primary"], "diff_ci_low": lo, "diff_ci_high": hi,
                         "p_melhor": float(np.mean(fin > 0)) if len(fin) else np.nan, "margem": ev.kpi.ni_margin,
                         "decisao": sc.paired_decision(lo, hi, ev.kpi.ni_margin)})
    return pd.DataFrame(rows)


def aggregate_quality(cfg: BenchmarkConfig, evals: dict[str, KPIEvaluation], backends: list[str], baseline: str,
                      kpis: list[str], weights: dict[str, float] | None = None) -> pd.DataFrame:
    """Qualidade normalizada média nos KPIs comuns (IC bootstrap pareado) e diferença contra a base, em pp."""
    w = np.array([(weights or {}).get(k, 1.0) for k in kpis], float)
    reps, points = {}, {}
    for b in backends:
        stack, pts = [], []
        for k in kpis:
            ev = evals[k]
            floor, ceiling = quality_bounds(cfg, ev)
            stack.append(normalize_quality(ev.dists[b], floor, ceiling))
            pts.append(normalize_quality(ev.table.set_index("model").at[b, "primary"], floor, ceiling))
        if not kpis:
            continue
        reps[b] = np.ma.average(np.ma.masked_invalid(np.vstack(stack)), axis=0, weights=w).filled(np.nan)
        points[b] = float(np.average(pts, weights=w))
    rows = []
    for b in backends:
        if b not in reps:
            rows.append({"backend": b})
            continue
        lo, hi = percentile_ci(reps[b], cfg.statistics.ci)
        row = {"backend": b, "q_comum": points[b], "q_comum_ci_low": lo, "q_comum_ci_high": hi}
        if baseline in reps:
            d = reps[b] - reps[baseline]
            dlo, dhi = percentile_ci(d, cfg.statistics.ci)
            fin = d[np.isfinite(d)]
            row.update(diff_base_pp=100 * (points[b] - points[baseline]), diff_base_ci_low_pp=100 * dlo,
                       diff_base_ci_high_pp=100 * dhi,
                       p_melhor_que_base=float(np.mean(fin > 0)) if b != baseline and len(fin) else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def calibration_table(cfg: BenchmarkConfig, preds: pd.DataFrame, ds, kpis) -> pd.DataFrame:
    """Confiança informada × acerto contra a referência (KPIs estruturados), por faixa de confiança."""
    p = preds[preds["confidence"].notna() & preds["valid"]] if "confidence" in preds else pd.DataFrame()
    if p.empty:
        return pd.DataFrame()
    gt_col = {k.name: k.gt_column for k in kpis if k.type != "text"}
    ref = ds.df.set_index(ds.id_col)
    p = p[p["kpi"].isin(gt_col)].copy()
    p["gt"] = [ref.at[c, gt_col[k]] if gt_col[k] in ref.columns else None for c, k in zip(p["call_id"], p["kpi"])]
    p = p[p["gt"].notna()]
    p["acerto"] = [str(v) == str(g) if not isinstance(g, (int, np.integer)) else int(float(v)) == int(g)
                   for v, g in zip(p["value"], p["gt"])]
    # só a confiança "própria" do backend (no híbrido, KPIs reextraídos guardam a confiança do Jev)
    p = p[~p["source"].astype(str).str.contains("fallback")]
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0001]
    p["faixa"] = pd.cut(p["confidence"].astype(float), bins, labels=["0–20%", "20–40%", "40–60%", "60–80%", "80–100%"],
                        right=False)
    g = p.groupby(["model", "faixa"], observed=False).agg(n=("acerto", "size"), acerto=("acerto", "mean")).reset_index()
    g = g.rename(columns={"model": "backend"})
    # curva de limiar: acima de t, qual a cobertura e o acerto (base do confidence_threshold do híbrido)
    curve = []
    for b, d in p.groupby("model"):
        for t in np.round(np.arange(0, 0.95, 0.05), 2):
            keep = d[d["confidence"].astype(float) >= t]
            curve.append({"backend": b, "limiar": t, "cobertura": len(keep) / len(d),
                          "acerto_acima": float(keep["acerto"].mean()) if len(keep) else np.nan,
                          "acerto_abaixo": float(d.loc[d["confidence"].astype(float) < t, "acerto"].mean())
                          if (d["confidence"].astype(float) < t).any() else np.nan})
    g["tipo_linha"] = "faixa"
    curve_df = pd.DataFrame(curve).assign(tipo_linha="limiar")
    return pd.concat([g, curve_df], ignore_index=True)


# =============================================================================== custo e latência
def _projected(ctx: X.ExtractorContext, reqs: pd.DataFrame) -> pd.Series:
    pj = ctx.cfg.projection

    def one(r):
        spec = ctx.cfg.model(r["model_id"])
        tin = r["input_tokens"] + r["cached_tokens"] + r["cache_write_tokens"]
        return projected_cost_per_call(spec, tin, r["output_tokens"], r["prefix_tokens"], pj.cache_hit_ratio,
                                       pj.use_batch)
    ok = reqs["latency_s"].notna()
    return reqs[ok].apply(one, axis=1) if ok.any() else pd.Series(dtype=float)


def summary_table(ctx: X.ExtractorContext, per_call: pd.DataFrame, reqs: pd.DataFrame, backends: list[str],
                  kinds: dict, coverage: dict, n_kpis: int, sla_s: float | None) -> pd.DataFrame:
    pj = ctx.cfg.projection
    r = reqs.copy()
    r["proj_usd"] = _projected(ctx, r)
    rows = []
    for b in backends:
        g = per_call[per_call["backend"] == b]
        n = len(g)
        rb = r[r["backend"] == b]
        proj = float(rb["proj_usd"].sum() / n) if n else np.nan
        ok = g[g["latency_s"].notna()]
        tin = g["input_tokens"] + g["cached_tokens"] + g["cache_write_tokens"]
        models = sorted(set(rb["model_id"]))
        row = {"backend": b, "tipo": kinds[b], "descricao": KIND_LABELS.get(kinds[b], kinds[b]), "modelos": ", ".join(models),
               "kpis_cobertos": len(coverage[b]), "cobertura_pct": len(coverage[b]) / n_kpis, "ligacoes": n,
               "requisicoes_por_ligacao": float(g["requisicoes"].mean()),
               "entrada_tokens": float(tin.mean()), "cache_tokens": float(g["cached_tokens"].mean()),
               "saida_tokens": float(g["output_tokens"].mean()), "raciocinio_tokens": float(g["reasoning_tokens"].mean()),
               "total_tokens": float((tin + g["output_tokens"]).mean()),
               "usd_por_ligacao": float(g["cost_usd"].mean()), "usd_1k_medido": float(g["cost_usd"].mean() * 1000),
               "usd_1k_projetado": proj * 1000, "usd_mes_projetado": proj * pj.monthly_calls,
               "usd_ano_projetado": proj * pj.monthly_calls * 12,
               "latencia_p50_s": float(ok["latency_s"].median()) if len(ok) else np.nan,
               "latencia_p95_s": float(ok["latency_s"].quantile(0.95)) if len(ok) else np.nan,
               "latencia_p99_s": float(ok["latency_s"].quantile(0.99)) if len(ok) else np.nan,
               "dentro_sla_pct": float((ok["latency_s"] <= sla_s).mean()) if sla_s and len(ok) else np.nan,
               "kpis_validos_pct": float(g["kpis_validos"].sum() / g["n_kpis"].sum()) if g["n_kpis"].sum() else np.nan,
               "ligacoes_completas_pct": float((g["kpis_validos"] == g["n_kpis"]).mean()) if n else np.nan,
               "falhas_requisicao_pct": float(rb["error"].notna().mean()) if len(rb) else np.nan,
               "truncadas_pct": float(g["truncado"].mean()) if n else np.nan,
               "ligacoes_com_fallback_pct": float((g["fallback_kpis"] > 0).mean()) if n else np.nan,
               "kpis_fallback_por_ligacao": float(g["fallback_kpis"].mean()) if n else np.nan,
               "preco_estimado": any(ctx.cfg.model(m).pricing_is_estimate for m in models),
               "simulado": bool(g["simulated"].any())}
        rows.append(row)
    df = pd.DataFrame(rows)
    return df


def cost_breakdown(ctx: X.ExtractorContext, reqs: pd.DataFrame, per_call: pd.DataFrame) -> pd.DataFrame:
    """Custo medido por componente: tokens médios por ligação × preço por 1M (soma = custo medido)."""
    rows = []
    for (b, m), g in reqs[reqs["latency_s"].notna()].groupby(["backend", "model_id"], sort=False):
        n = int((per_call["backend"] == b).sum())
        if not n:
            continue
        p = ctx.cfg.model(m).pricing
        price = {"input": p.input, "cached_input": p.cached_input, "cache_write": p.input * p.cache_write_multiplier,
                 "output": p.output}
        tok = {c: float(g[c].sum() / n) for c in ("input_tokens", "cached_tokens", "cache_write_tokens", "reasoning_tokens")}
        tok["saida_visivel_tokens"] = float((g["output_tokens"] - g["reasoning_tokens"]).clip(lower=0).sum() / n)
        for col, label, key in COST_COMPONENTS:
            usd = tok[col] * price[key] / 1e6
            rows.append({"backend": b, "modelo": m, "componente": label, "tokens_por_ligacao": tok[col],
                         "preco_usd_por_1M": price[key], "usd_por_ligacao": usd, "usd_1k_ligacoes": usd * 1000,
                         "requisicoes_por_ligacao": len(g) / n})
    df = pd.DataFrame(rows)
    if not df.empty:
        tot = df.groupby("backend")["usd_1k_ligacoes"].transform("sum")
        df["pct_do_custo"] = np.where(tot > 0, 100 * df["usd_1k_ligacoes"] / tot, np.nan)
    return df


# =============================================================================== decisão
def recommend(summary: pd.DataFrame, baseline: str, n_kpis: int) -> dict:
    """Regra: para SUBSTITUIR o sistema atual o backend precisa cobrir todos os KPIs e não ser inferior à base
    na qualidade dos KPIs comuns (decisão melhora/equivalente/não-inferior); entre os elegíveis, vence o menor
    custo projetado, com a latência como desempate. Também aponta o melhor por dimensão."""
    s = summary.set_index("backend")
    ok_q = s["decisao_vs_base"].isin(["base", "melhora", "equivalente", "nao_inferior"])
    full = s["kpis_cobertos"] == n_kpis
    eligible = s[ok_q & full].sort_values(["usd_1k_projetado", "latencia_p50_s"])
    best_q = s["q_comum"].idxmax() if s["q_comum"].notna().any() else None
    rec = {
        "recomendado": eligible.index[0] if len(eligible) else baseline,
        "elegiveis": list(eligible.index),
        "mais_barato": s["usd_1k_projetado"].idxmin(),
        "mais_rapido": s["latencia_p50_s"].idxmin(),
        "melhor_qualidade_kpis_comuns": best_q,
        "menor_custo_por_pp": s["usd_1k_por_pp"].idxmin() if s["usd_1k_por_pp"].notna().any() else None,
        "fronteira": list(s[s["status_ce"] == "fronteira"].sort_values("usd_1k_projetado").index),
        "dominados": list(s[s["status_ce"].astype(str).str.startswith("dominado")].index),
        "cobertura_parcial": list(s[~full].index),
        "nao_elegiveis_por_qualidade": list(s[full & ~ok_q].index),
    }
    if rec["recomendado"] == baseline and not len(eligible.drop(baseline, errors="ignore")):
        rec["motivo"] = ("nenhum backend com cobertura total ficou não-inferior à base com esta amostra; "
                         "mantenha o sistema atual ou aumente a amostra")
    else:
        r = s.loc[rec["recomendado"]]
        rec["motivo"] = (f"cobre todos os KPIs, qualidade {r['decisao_vs_base']} frente à base e menor custo projetado "
                         "entre os elegíveis")
    return rec


# =============================================================================== relatório
def final_report(st: BackendStudy) -> str:
    from kpibench.reporting.aa_charts import br, usd

    s = st.summary.set_index("backend")
    base = s.loc[st.baseline]
    rec = st.recommendation
    pj = st.cfg.projection
    pct = lambda v: "—" if v is None or not np.isfinite(v) else f"{br(v, 1)}%"  # noqa: E731
    fine = lambda v: f"US$ {br(v, 6)}"  # noqa: E731
    sgn = lambda v: ("+" if v >= 0 else "−") + br(abs(v), 1)  # noqa: E731

    def md(df):
        return df.to_markdown(index=False, disable_numparse=True)

    L = ["# Relatório: LLM × CDM × Jev na extração de KPIs", ""]
    if st.simulated:
        L += ["> ⚠️ **RODADA SIMULADA.** Respostas geradas pelo provider mock (CDM e Jev com perfis fictícios); "
              "tokens dos prompts e payloads reais. Valida o fluxo, não serve para decidir.", ""]
    L += [f"**{int(base['ligacoes'])} ligações** · base: `{st.baseline}` · referência: `{st.reference}` "
          f"({st.reference_mode}) · KPIs comuns a todos os backends: {len(st.common_kpis)} de "
          f"{int(s['kpis_cobertos'].max())} · margem de qualidade: {br(st.quality_margin_pp, 1)} pp.", ""]

    # 1. veredito
    r = s.loc[rec["recomendado"]]
    L += ["## 1. Qual compensa mais", "",
          f"**Recomendado para substituir o sistema atual: `{rec['recomendado']}`** ({r['descricao']}) — "
          f"{rec['motivo']}. Custo projetado {usd(r['usd_1k_projetado'])}/1.000 ligações "
          f"({pct(100 * (r['usd_1k_projetado'] / base['usd_1k_projetado'] - 1))} frente à base), latência p50 "
          f"{br(r['latencia_p50_s'], 2)} s, qualidade nos KPIs comuns {pct(100 * r['q_comum'])}.", "",
          f"- **Mais barato:** `{rec['mais_barato']}` · **mais rápido:** `{rec['mais_rapido']}` · **melhor qualidade "
          f"(KPIs comuns):** `{rec['melhor_qualidade_kpis_comuns']}` · **menor custo por ponto de qualidade:** "
          f"`{rec['menor_custo_por_pp']}`.",
          f"- **Fronteira de custo-efetividade** (projetado × qualidade nos KPIs comuns): "
          + " → ".join(f"`{b}`" for b in rec["fronteira"]) + ". "
          + (f"Dominados (há opção mais barata e melhor): {', '.join('`' + b + '`' for b in rec['dominados'])}."
             if rec["dominados"] else "")]
    if rec["cobertura_parcial"]:
        L.append(f"- **Cobertura parcial** (não substituem o sistema sozinhos): "
                 + ", ".join(f"`{b}` ({int(s.at[b, 'kpis_cobertos'])}/{int(s['kpis_cobertos'].max())} KPIs)"
                             for b in rec["cobertura_parcial"])
                 + ". O Jev não gera texto livre; os KPIs de justificativa precisam de LLM/CDM (backend híbrido).")
    if rec["nao_elegiveis_por_qualidade"]:
        L.append(f"- **Fora por qualidade** (inferiores ou inconclusivos frente à base): "
                 + ", ".join(f"`{b}` ({s.at[b, 'decisao_vs_base']})" for b in rec["nao_elegiveis_por_qualidade"]) + ".")
    L.append("")

    # 2. custo
    L += ["## 2. Custo", "",
          f"Medido = `usage` × preço nesta amostra; projetado = produção ({br(pj.monthly_calls, 0)} ligações/mês, "
          f"{'Batch' if pj.use_batch else 'síncrono'}, cache hit {pj.cache_hit_ratio:.0%} no prefixo cacheável).", ""]
    t = pd.DataFrame([{"backend": b, "requisições/ligação": br(x["requisicoes_por_ligacao"], 1),
                       "US$/ligação": fine(x["usd_por_ligacao"]), "US$/1k medido": usd(x["usd_1k_medido"]),
                       "US$/1k projetado": usd(x["usd_1k_projetado"]), "US$/mês": usd(x["usd_mes_projetado"]),
                       "US$/ano": usd(x["usd_ano_projetado"]),
                       "vs base": pct(100 * (x["usd_1k_projetado"] / base["usd_1k_projetado"] - 1)),
                       "preço": "estimado" if x["preco_estimado"] else "tabela"} for b, x in s.iterrows()])
    L += [md(t), ""]

    # 3. latência
    L += ["## 3. Latência por ligação", "",
          "Tempo para obter todos os KPIs de uma ligação (no sistema atual, as chamadas por KPI em paralelo; no "
          "híbrido, Jev e depois CDM em sequência).", ""]
    t = pd.DataFrame([{"backend": b, "p50 (s)": br(x["latencia_p50_s"], 2), "p95 (s)": br(x["latencia_p95_s"], 2),
                       "p99 (s)": br(x["latencia_p99_s"], 2),
                       **({f"≤ SLA {br(st.sla_s, 1)} s": pct(100 * x["dentro_sla_pct"])} if st.sla_s else {}),
                       "× base (p50)": br(x["latencia_p50_s"] / base["latencia_p50_s"], 2) + "×"} for b, x in s.iterrows()])
    L += [md(t), ""]

    # 4. qualidade
    L += ["## 4. Qualidade", "",
          f"Qualidade normalizada (0% = acaso, 100% = concordância total com a referência), média dos "
          f"{len(st.common_kpis)} KPIs que todos cobrem ({', '.join(st.common_kpis)}), IC 95% bootstrap pareado. "
          f"Decisão contra a base com margem de {br(st.quality_margin_pp, 1)} pp.", ""]
    t = pd.DataFrame([{"backend": b, "qualidade (KPIs comuns)": f"{pct(100 * x['q_comum'])} [{br(100 * x['q_comum_ci_low'], 1)}; "
                                                                f"{br(100 * x['q_comum_ci_high'], 1)}]",
                       "Δ vs base (pp)": "—" if b == st.baseline else
                       f"{sgn(x['diff_base_pp'])} [{br(x['diff_base_ci_low_pp'], 1)}; {br(x['diff_base_ci_high_pp'], 1)}]",
                       "decisão": x["decisao_vs_base"], "cobertura": f"{int(x['kpis_cobertos'])} KPIs",
                       "KPIs válidos": pct(100 * x["kpis_validos_pct"])} for b, x in s.iterrows()])
    L += [md(t), ""]
    if not st.comparisons.empty:
        cnt = st.comparisons.pivot_table(index="backend", columns="decisao", values="kpi", aggfunc="count", fill_value=0)
        L += ["Decisão por KPI contra a base (contagem de KPIs):", "", cnt.reset_index().to_markdown(index=False), ""]

    # 5. qualidade × custo
    L += ["## 5. Relação qualidade × custo", "",
          "Fronteira de eficiência com custo projetado e qualidade nos KPIs comuns: dominado = existe opção mais "
          "barata e igual ou melhor; ICER = US$ por 1.000 ligações para cada ponto percentual de qualidade a mais "
          "frente ao vizinho anterior da fronteira.", ""]
    t = pd.DataFrame([{"backend": b, "US$/1k projetado": usd(x["usd_1k_projetado"]), "qualidade": pct(100 * x["q_comum"]),
                       "US$/1k por pp": br(x["usd_1k_por_pp"], 5), "status": x["status_ce"],
                       "ICER (US$/1k por pp)": br(x["icer_usd_1k_por_pp"], 5) if np.isfinite(x["icer_usd_1k_por_pp"]) else "—",
                       "vizinho anterior": x["vizinho_anterior"] or "—"}
                      for b, x in s.sort_values("usd_1k_projetado").iterrows()])
    L += [md(t), ""]

    # 6. cálculo
    L += ["## 6. Cálculo do custo e consumo de tokens", "",
          "Custo por ligação = Σ por requisição (tokens × preço do componente por 1M) ÷ 1.000.000. O Jev cobra só a "
          "entrada; raciocínio de LLM é cobrado como saída.", ""]
    bd = st.breakdown
    for b in st.backends:
        d = bd[(bd["backend"] == b) & (bd["tokens_por_ligacao"] > 0)]
        if d.empty:
            continue
        terms = " + ".join(f"{br(x.tokens_por_ligacao, 0)} × US$ {br(x.preco_usd_por_1M, 3)} ({x.componente}, {x.modelo})"
                           for x in d.itertuples())
        L.append(f"- **`{b}`:** ({terms}) ÷ 1.000.000 = **{fine(d['usd_por_ligacao'].sum())} por ligação** = "
                 f"{usd(d['usd_1k_ligacoes'].sum())}/1.000.")
    L.append("")
    t = pd.DataFrame([{"backend": b, "requisições": br(x["requisicoes_por_ligacao"], 1), "entrada": br(x["entrada_tokens"], 0),
                       "cache": br(x["cache_tokens"], 0), "saída": br(x["saida_tokens"], 0),
                       "raciocínio": br(x["raciocinio_tokens"], 0), "total": br(x["total_tokens"], 0),
                       "vs base": pct(100 * (x["total_tokens"] / base["total_tokens"] - 1))} for b, x in s.iterrows()])
    L += ["Tokens por ligação:", "", md(t), ""]

    # 7. confiabilidade
    L += ["## 7. Confiabilidade e operação", ""]
    for b, x in s.iterrows():
        extra = ""
        if str(x["tipo"]).startswith("hybrid"):
            extra = (f"; fallback em {pct(100 * x['ligacoes_com_fallback_pct'])} das ligações "
                     f"({br(x['kpis_fallback_por_ligacao'], 2)} KPIs reextraídos por ligação)")
        L.append(f"- `{b}`: {pct(100 * x['ligacoes_completas_pct'])} das ligações com todos os KPIs válidos; "
                 f"{pct(100 * x['falhas_requisicao_pct'])} de requisições com erro; {pct(100 * x['truncadas_pct'])} "
                 f"truncadas{extra}.")
    cal = st.calibration
    if not cal.empty and (cal["tipo_linha"] == "faixa").any():
        f = cal[(cal["tipo_linha"] == "faixa") & (cal["n"] > 0)]
        for b, d in f.groupby("backend"):
            L.append(f"- Calibração de `{b}` (acerto contra a referência por faixa de confiança): "
                     + "; ".join(f"{x.faixa}: {pct(100 * x.acerto)} (n={int(x.n)})" for x in d.itertuples()) + ".")
    L.append("")

    # 8. ressalvas
    L += ["## 8. Ressalvas", "",
          "- **Preços:** os de CDM e Jev vêm de `config/extractors.yaml`; os marcados como estimados precisam ser "
          "confirmados no contrato antes da decisão.",
          "- **Jev** decide entre opções e escalas; não explica nem gera texto. A qualidade nos KPIs de texto vem do "
          "secundário do híbrido.",
          "- **CDM ScaleDown**: Decisions (choice/score) nos KPIs estruturados e Summarize nos de texto; cada pergunta "
          "da Decisions API é cobrada como uma chamada e o custo medido usa o `usage.cost` informado pela API.",
          "- **Decodificação restrita** (CDM, structured outputs) garante a estrutura (valores no domínio), não a "
          "correção semântica: o ganho esperado é em confiabilidade e custo; a qualidade depende do modelo.",
          "- **Amostra:** ICs largos com poucas dezenas de ligações; decisões 'inconclusivo' pedem mais ligações."]
    L += [f"- {n}" for n in st.notes]
    return "\n".join(L)


# =============================================================================== produção
def production_snippet(backend: str) -> str:
    return (f"python -m kpibench extract --backend {backend} --input ligacoes.csv --output kpis_{backend}.jsonl "
            "--config config/benchmark.yaml")
