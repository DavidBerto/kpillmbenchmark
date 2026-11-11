"""Custo-efetividade: ICER (razão custo-efetividade incremental) na fronteira de eficiência.

Por que não `qualidade / custo`: a razão explode quando o custo tende a zero (modelo ruim e
quase grátis "vence"), o zero das métricas é arbitrário (QWK pode ser negativo; macro-F1 = 0
não é o acaso) e ignora respostas inválidas. Aqui, como em avaliação econômica em saúde:

  1. qualidade normalizada  q̃ = (primária − piso) / (teto − piso) ∈ [0, 1], com piso = acaso
     (0 para QWK/juiz; 1/K para macro-F1 — F1 esperado de um preditor aleatório que segue a
     distribuição das K classes presentes);
  2. custo efetivo          C_ef = custo projetado por 1k chamadas / taxa de respostas válidas
     (custo por 1.000 respostas aproveitáveis);
  3. âncora implícita "acaso" (custo 0, q̃ 0): modelo que não supera o acaso é dominado por ela;
  4. dominância forte (custa mais e entrega q̃ menor ou igual) e estendida (ICER não crescente:
     o ponto do meio é superado por uma combinação dos vizinhos);
  5. ICER = ΔC_ef / Δq̃ entre vizinhos da fronteira, em US$ por 1.000 chamadas por ponto
     percentual (pp) de qualidade normalizada.

A incerteza vem das mesmas réplicas bootstrap pareadas do teste de não-inferioridade:
`p_fronteira` = fração das réplicas em que o modelo fica na fronteira.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from kpibench.config import BenchmarkConfig
from kpibench.metrics.kpi_eval import BASELINE_NAME, KPIEvaluation

ANCHOR = "acaso"
FRONTIER, DOMINATED, EXT_DOMINATED, NO_DATA = "fronteira", "dominado", "dominado_estendido", "sem_dados"
CE_COLUMNS = ["q_norm", "usd_1k_efetivo", "status_ce", "icer_usd_1k_por_pp", "vizinho_anterior", "p_fronteira"]


def normalize_quality(x, floor: float, ceiling: float):
    """q̃ = (x − piso) / (teto − piso), recortado em [0, 1]. NaN se teto <= piso."""
    if not ceiling > floor:
        return np.full_like(np.asarray(x, dtype=float), np.nan) if np.ndim(x) else float("nan")
    q = np.clip((np.asarray(x, dtype=float) - floor) / (ceiling - floor), 0.0, 1.0)
    return q if np.ndim(x) else float(q)


def quality_bounds(cfg: BenchmarkConfig, ev: KPIEvaluation) -> tuple[float, float]:
    """(piso, teto) do KPI: override em kpis.yaml > cost_effectiveness do benchmark; 'chance' = ev.chance."""
    ce = cfg.cost_effectiveness
    floor = ev.kpi.quality_floor if ev.kpi.quality_floor is not None else ce.quality_floor
    ceiling = ev.kpi.quality_ceiling if ev.kpi.quality_ceiling is not None else ce.quality_ceiling
    return (ev.chance if floor == "chance" else float(floor)), float(ceiling)


def effective_cost(cost_per_1k, valid_rate):
    """Custo por 1.000 respostas válidas; sem nenhuma resposta válida -> NaN (sem dados)."""
    c = np.asarray(cost_per_1k, dtype=float)
    v = np.asarray(valid_rate, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(v > 0, c / v, np.nan)


def _icer(c0: float, q0: float, c1: float, q1: float) -> float:
    return (c1 - c0) / ((q1 - q0) * 100)


def frontier_core(cost: np.ndarray, q: np.ndarray) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Fronteira de eficiência com âncora (0, 0).

    Retorna (status, icer, anterior) por posição; `anterior` = índice do vizinho anterior na
    fronteira, -1 para a âncora "acaso" e -2 para quem não está na fronteira.
    """
    cost, q = np.asarray(cost, float), np.asarray(q, float)
    n = len(cost)
    status = [NO_DATA] * n
    icer = np.full(n, np.nan)
    prev = np.full(n, -2, dtype=int)
    ok = np.flatnonzero(np.isfinite(cost) & np.isfinite(q))
    # custo crescente; no empate de custo, maior qualidade primeiro
    order = ok[np.lexsort((-q[ok], cost[ok]))]
    front, best = [], 0.0
    for i in order:
        if q[i] > best:
            front.append(int(i))
            best = q[i]
        else:
            status[i] = DOMINATED
    # dominância estendida: os ICERs ao longo da fronteira precisam ser crescentes
    while True:
        pts = [(0.0, 0.0)] + [(cost[i], q[i]) for i in front]
        icers = [_icer(*pts[j], *pts[j + 1]) for j in range(len(front))]
        bad = next((j for j in range(len(icers) - 1) if icers[j] > icers[j + 1]), None)
        if bad is None:
            break
        status[front.pop(bad)] = EXT_DOMINATED
    for k, i in enumerate(front):
        status[i] = FRONTIER
        icer[i] = icers[k]
        prev[i] = front[k - 1] if k else -1
    return status, icer, prev


def frontier_mask(cost: np.ndarray, q: np.ndarray) -> np.ndarray:
    status, _, _ = frontier_core(cost, q)
    return np.array([s == FRONTIER for s in status])


def frontier_icer(df: pd.DataFrame, cost_col: str = "usd_1k_efetivo", q_col: str = "q_norm",
                  model_col: str = "model") -> pd.DataFrame:
    """Anota status_ce, icer_usd_1k_por_pp e vizinho_anterior num grupo (um KPI ou o agregado)."""
    out = df.reset_index(drop=True).copy()
    status, icer, prev = frontier_core(out[cost_col].to_numpy(), out[q_col].to_numpy())
    models = out[model_col].tolist()
    out["status_ce"] = status
    out["icer_usd_1k_por_pp"] = icer
    out["vizinho_anterior"] = [ANCHOR if p == -1 else models[p] if p >= 0 else None for p in prev]
    return out


def _p_frontier(cost: np.ndarray, q_reps: np.ndarray) -> np.ndarray:
    """q_reps: (modelos, réplicas). Fração das réplicas em que cada modelo está na fronteira."""
    if q_reps.size == 0:
        return np.full(len(cost), np.nan)
    hits = np.zeros(len(cost))
    for b in range(q_reps.shape[1]):
        hits += frontier_mask(cost, q_reps[:, b])
    return hits / q_reps.shape[1]


def aggregate_point(per_kpi: pd.DataFrame, weights: dict[str, float] | None = None) -> pd.DataFrame:
    """Um ponto por modelo: soma do custo efetivo nos KPIs (= US$ por 1.000 ligações com todos os KPIs)
    e média de q̃ ponderada por KPI; depois a mesma fronteira/ICER."""
    d = per_kpi[per_kpi["model"] != BASELINE_NAME].dropna(subset=["q_norm", "usd_1k_efetivo"]).copy()
    if d.empty:
        return pd.DataFrame(columns=["model", "n_kpis", "q_norm", "usd_1k_efetivo", "status_ce",
                                     "icer_usd_1k_por_pp", "vizinho_anterior"])
    d["w"] = d["kpi"].map(lambda k: (weights or {}).get(k, 1.0))
    agg = d.groupby("model").apply(lambda g: pd.Series({
        "n_kpis": g["kpi"].nunique(),
        "q_norm": np.average(g["q_norm"], weights=g["w"]) if g["w"].sum() > 0 else np.nan,
        "usd_1k_efetivo": g["usd_1k_efetivo"].sum(),
    }), include_groups=False).reset_index()
    # comparar só modelos avaliados no mesmo conjunto de KPIs
    full_set = agg["n_kpis"] == agg["n_kpis"].max()
    agg.loc[~full_set, ["q_norm", "usd_1k_efetivo"]] = np.nan
    return frontier_icer(agg)


def cost_effectiveness(cfg: BenchmarkConfig, evals: dict[str, KPIEvaluation], costs: pd.DataFrame
                       ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retorna (por_kpi_modelo, agregado_por_modelo) com q̃, custo efetivo, status, ICER e p_fronteira."""
    weights = cfg.cost_effectiveness.kpi_weights
    per_kpi, reps = [], {}
    for name, ev in evals.items():
        floor, ceiling = quality_bounds(cfg, ev)
        t = ev.table.loc[ev.table["model"] != BASELINE_NAME, ["model", "primary", "valid_rate"]]
        t = t.merge(costs.loc[costs["kpi"] == name, ["model", "proj_usd_per_1k"]], on="model", how="left")
        t["q_norm"] = normalize_quality(t["primary"].to_numpy(), floor, ceiling)
        t["usd_1k_efetivo"] = effective_cost(t["proj_usd_per_1k"], t["valid_rate"])
        t = frontier_icer(t)
        q_reps = np.vstack([normalize_quality(ev.dists[m], floor, ceiling) for m in t["model"]]) \
            if len(t) else np.empty((0, 0))
        t["p_fronteira"] = _p_frontier(t["usd_1k_efetivo"].to_numpy(), q_reps)
        t.insert(0, "kpi", name)
        t["piso"], t["teto"] = floor, ceiling
        per_kpi.append(t)
        reps[name] = dict(zip(t["model"], q_reps))
    per = pd.concat(per_kpi, ignore_index=True) if per_kpi else pd.DataFrame(columns=["kpi", "model", *CE_COLUMNS])

    agg = aggregate_point(per, weights)
    if not agg.empty:
        models = agg["model"].tolist()
        rows = []
        for m in models:
            ks = [k for k in reps if m in reps[k]]
            w = np.array([weights.get(k, 1.0) for k in ks])
            rows.append(np.average(np.vstack([reps[k][m] for k in ks]), axis=0, weights=w)
                        if len(ks) and w.sum() > 0 else np.full(cfg.statistics.n_bootstrap, np.nan))
        agg["p_fronteira"] = _p_frontier(agg["usd_1k_efetivo"].to_numpy(), np.vstack(rows))
    return per, agg
