"""Fronteira de Pareto custo x performance e recomendação de roteamento por KPI.

Regra de decisão (por KPI):
  1. referência = candidato com maior métrica primária (ponto);
  2. elegíveis  = referência + modelos NÃO-INFERIORES a ela (IC bootstrap pareado da
     diferença abaixo da margem `ni_margin` do KPI);
  3. recomendado = elegível com menor custo projetado por chamada.
Modelos "inconclusivos" NÃO são elegíveis: falta amostra para afirmar empate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from kpibench.config import BenchmarkConfig
from kpibench.metrics.kpi_eval import BASELINE_NAME, KPIEvaluation
from kpibench.metrics.stats import compare_to_reference


def pareto_front(points: pd.DataFrame, cost_col: str, perf_col: str) -> set[str]:
    p = points.dropna(subset=[cost_col, perf_col]).sort_values([cost_col, perf_col], ascending=[True, False])
    front, best = set(), -np.inf
    for r in p.itertuples():
        perf = getattr(r, perf_col)
        if perf > best:
            front.add(r.model)
            best = perf
    return front


def comparisons_table(cfg: BenchmarkConfig, ev: KPIEvaluation) -> tuple[pd.DataFrame, str]:
    cand = ev.table[ev.table["model"] != BASELINE_NAME]
    ref = str(cand.sort_values("primary", ascending=False).iloc[0]["model"])
    point = dict(zip(ev.table["model"], ev.table["primary"]))
    comps = compare_to_reference(point, ev.dists, ref, ev.kpi.ni_margin, cfg.statistics.ci)
    df = pd.DataFrame([c.__dict__ for c in comps])
    df.insert(0, "kpi", ev.kpi.name)
    return df, ref


def routing(cfg: BenchmarkConfig, evals: dict[str, KPIEvaluation], costs: pd.DataFrame
            ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Retorna (tabela_por_kpi_modelo, recomendação_por_kpi, comparação_de_planos)."""
    full_rows, rec_rows = [], []
    for name, ev in evals.items():
        comp, ref = comparisons_table(cfg, ev)
        t = ev.table.merge(comp[["model", "diff", "ci_low", "ci_high", "p_worse", "decision"]]
                           .rename(columns={"ci_low": "diff_ci_low", "ci_high": "diff_ci_high"}),
                           on="model", how="left")
        t = t.merge(costs[costs["kpi"] == name][["model", "proj_usd_per_1k", "proj_usd_month",
                                                  "latency_p50_s", "latency_p95_s", "tier"]],
                    on="model", how="left")
        front = pareto_front(t[t["model"] != BASELINE_NAME], "proj_usd_per_1k", "primary")
        t["pareto"] = t["model"].isin(front)
        t["perf_por_usd"] = t["primary"] / t["proj_usd_per_1k"]
        full_rows.append(t)

        elig = t[t["decision"].isin(["referencia", "nao_inferior"]) & (t["model"] != BASELINE_NAME)]
        chosen = elig.sort_values("proj_usd_per_1k").iloc[0]
        best = t[t["model"] == ref].iloc[0]
        base = t[t["model"] == BASELINE_NAME]
        # candidato econômico: não declarado inferior e ponto dentro da margem -> validar com mais amostra
        econ_pool = t[(t["model"] != BASELINE_NAME) & (t["decision"] != "inferior")
                      & (t["primary"] >= best["primary"] - ev.kpi.ni_margin)]
        econ = econ_pool.sort_values("proj_usd_per_1k").iloc[0]
        rec_rows.append({
            "kpi": name, "display": ev.kpi.display, "metrica": ev.primary_name, "n": ev.n_items,
            "recomendado": chosen["model"], "perf_recomendado": chosen["primary"],
            "ci_recomendado": f"[{chosen['ci_low']:.3f}, {chosen['ci_high']:.3f}]",
            "usd_1k_recomendado": chosen["proj_usd_per_1k"],
            "melhor": ref, "perf_melhor": best["primary"], "usd_1k_melhor": best["proj_usd_per_1k"],
            "economia_vs_melhor_pct": 100 * (1 - chosen["proj_usd_per_1k"] / best["proj_usd_per_1k"])
            if best["proj_usd_per_1k"] else 0.0,
            "perf_producao_atual": float(base["primary"].iloc[0]) if len(base) else np.nan,
            "n_inconclusivos": int((t["decision"] == "inconclusivo").sum()),
            "usd_mes_recomendado": chosen["proj_usd_month"],
            "candidato_economico": econ["model"], "perf_candidato": econ["primary"],
            "decisao_candidato": econ["decision"], "usd_1k_candidato": econ["proj_usd_per_1k"],
            "usd_mes_candidato": econ["proj_usd_month"],
            "margem_ni": ev.kpi.ni_margin,
        })

    full = pd.concat(full_rows, ignore_index=True)
    rec = pd.DataFrame(rec_rows)
    plans = _plans(cfg, full, rec)
    return full, rec, plans


def _plans(cfg: BenchmarkConfig, full: pd.DataFrame, rec: pd.DataFrame) -> pd.DataFrame:
    """Compara: plano roteado x um único modelo para todos os KPIs."""
    rows = [{
        "plano": "roteado_por_kpi (conservador)",
        "usd_mes": float(rec["usd_mes_recomendado"].sum()),
        "kpis_nao_inferiores": f"{len(rec)}/{len(rec)}",
        "perf_media": float(rec["perf_recomendado"].mean()),
    }, {
        "plano": "roteado_economico (a validar)",
        "usd_mes": float(rec["usd_mes_candidato"].sum()),
        "kpis_nao_inferiores": f"{int(rec['decisao_candidato'].isin(['referencia', 'nao_inferior']).sum())}/{len(rec)}",
        "perf_media": float(rec["perf_candidato"].mean()),
    }]
    cand = full[full["model"] != BASELINE_NAME]
    for m, g in cand.groupby("model"):
        ok = g["decision"].isin(["referencia", "nao_inferior"]).sum()
        rows.append({"plano": f"so_{m}", "usd_mes": float(g["proj_usd_month"].sum()),
                     "kpis_nao_inferiores": f"{ok}/{len(g)}", "perf_media": float(g["primary"].mean())})
    df = pd.DataFrame(rows).sort_values("usd_mes").reset_index(drop=True)
    return df
