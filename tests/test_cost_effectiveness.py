"""Custo-efetividade: normalização, custo efetivo, fronteira/ICER e robustez bootstrap."""
import numpy as np
import pandas as pd
import pytest

from kpibench.analysis.cost_effectiveness import (ANCHOR, DOMINATED, EXT_DOMINATED, FRONTIER, NO_DATA,
                                                  _p_frontier, aggregate_point, effective_cost, frontier_icer,
                                                  normalize_quality)
from kpibench.config import PROJECT_ROOT, load_config
from kpibench.data.loader import load_dataset
from kpibench.metrics.kpi_eval import evaluate_kpi


def _frame(points: dict[str, tuple[float, float]]) -> pd.DataFrame:
    return pd.DataFrame([{"model": m, "usd_1k_efetivo": c, "q_norm": q} for m, (c, q) in points.items()])


def test_hand_computed_frontier_with_strong_and_extended_dominance():
    t = frontier_icer(_frame({"A": (1, .5), "B": (2, .4), "C": (3, .6), "D": (5, .9)})).set_index("model")
    assert t.loc["B", "status_ce"] == DOMINATED              # custa mais que A e entrega menos
    assert t.loc["C", "status_ce"] == EXT_DOMINATED          # ICER A→C (0,2) > C→D (0,067)
    assert t.loc["A", "status_ce"] == FRONTIER and t.loc["D", "status_ce"] == FRONTIER
    assert t.loc["A", "vizinho_anterior"] == ANCHOR
    assert t.loc["A", "icer_usd_1k_por_pp"] == pytest.approx(1 / 50)
    assert t.loc["D", "vizinho_anterior"] == "A"
    assert t.loc["D", "icer_usd_1k_por_pp"] == pytest.approx(4 / 40)
    assert np.isnan(t.loc["B", "icer_usd_1k_por_pp"]) and t.loc["B", "vizinho_anterior"] is None


def test_cheap_model_at_chance_level_is_dominated_where_naive_ratio_rewards_it():
    pts = {"quase_gratis": (0.001, 0.0), "bom": (2.0, 0.8)}
    naive = {m: q / c for m, (c, q) in {"quase_gratis": (0.001, 0.3), "bom": (2.0, 0.85)}.items()}
    assert max(naive, key=naive.get) == "quase_gratis"       # QWK 0,30 "vence" na razão simples
    t = frontier_icer(_frame(pts)).set_index("model")
    assert t.loc["quase_gratis", "status_ce"] == DOMINATED   # o acaso (custo zero) já entrega q̃ = 0
    assert t.loc["bom", "status_ce"] == FRONTIER


def test_icers_are_increasing_along_frontier():
    rng = np.random.default_rng(0)
    for _ in range(50):
        df = _frame({f"m{i}": (rng.uniform(0.01, 10), rng.uniform(0, 1)) for i in range(8)})
        f = frontier_icer(df)
        f = f[f["status_ce"] == FRONTIER].sort_values("usd_1k_efetivo")
        assert (np.diff(f["icer_usd_1k_por_pp"]) >= 0).all()
        assert (np.diff(f["q_norm"]) > 0).all()


def test_tie_in_cost_keeps_only_best_and_missing_is_flagged():
    t = frontier_icer(_frame({"a": (1, .5), "b": (1, .7), "c": (np.nan, .9)})).set_index("model")
    assert t.loc["b", "status_ce"] == FRONTIER and t.loc["a", "status_ce"] == DOMINATED
    assert t.loc["c", "status_ce"] == NO_DATA


def test_normalization_and_effective_cost():
    assert normalize_quality(0.5, 0.0, 1.0) == pytest.approx(0.5)
    assert normalize_quality(-0.2, 0.0, 1.0) == 0.0          # QWK negativo = pior que o acaso -> 0
    assert normalize_quality(0.5, 1 / 3, 1.0) == pytest.approx(0.25)
    np.testing.assert_allclose(normalize_quality(np.array([0.0, 2.0]), 0.0, 1.0), [0.0, 1.0])
    assert np.isnan(normalize_quality(0.5, 1.0, 1.0))
    np.testing.assert_allclose(effective_cost([2.0, 2.0], [1.0, 0.5]), [2.0, 4.0])  # metade inválida dobra
    assert np.isnan(effective_cost([2.0], [0.0])[0])


def test_p_frontier_bounds_and_always_dominated_model():
    cost = np.array([1.0, 2.0, 3.0])
    reps = np.array([np.full(200, 0.5), np.full(200, 0.3), np.random.default_rng(1).uniform(0.4, 0.7, 200)])
    p = _p_frontier(cost, reps)
    assert ((p >= 0) & (p <= 1)).all()
    assert p[0] == 1.0 and p[1] == 0.0 and 0 < p[2] < 1


def test_aggregate_sums_cost_and_weights_quality():
    per = pd.DataFrame([
        {"kpi": "k1", "model": "a", "q_norm": 0.2, "usd_1k_efetivo": 1.0},
        {"kpi": "k2", "model": "a", "q_norm": 0.6, "usd_1k_efetivo": 1.0},
        {"kpi": "k1", "model": "b", "q_norm": 0.8, "usd_1k_efetivo": 3.0},
        {"kpi": "k2", "model": "b", "q_norm": 0.8, "usd_1k_efetivo": 3.0},
        {"kpi": "k1", "model": "producao_atual", "q_norm": 0.9, "usd_1k_efetivo": np.nan},
    ])
    agg = aggregate_point(per, {"k1": 3.0}).set_index("model")
    assert set(agg.index) == {"a", "b"}
    assert agg.loc["a", "usd_1k_efetivo"] == pytest.approx(2.0)
    assert agg.loc["a", "q_norm"] == pytest.approx((3 * 0.2 + 0.6) / 4)
    assert agg.loc["b", "icer_usd_1k_por_pp"] == pytest.approx(4.0 / ((0.8 - 0.3) * 100))


@pytest.mark.parametrize("gt, k", [(["a", "b", "c"] * 10, 3), (["a"] * 27 + ["b"] * 2 + ["c"], 3), (["a", "b"] * 5, 2)])
def test_macro_f1_chance_level(tmp_path, gt, k):
    cfg = load_config(PROJECT_ROOT / "config" / "demo.yaml")
    kpi = next(x for x in cfg.selected_kpis if x.type == "categorical")
    labels = list(kpi.labels)[:3]
    ids = [str(i) for i in range(len(gt))]
    df = pd.DataFrame({"call_id": ids, "resumo": "x", kpi.gt_column: [labels["abc".index(g)] for g in gt]})
    path = tmp_path / "d.csv"
    df.to_csv(path, index=False)
    cfg.data.path, cfg.data.baseline_prefix, cfg.data.transcript_column = str(path), None, None
    cfg.data.reference_summary_column = None
    cfg.statistics.n_bootstrap = 20
    preds = pd.DataFrame({"kpi": kpi.name, "model": "m", "call_id": ids, "value": df[kpi.gt_column], "valid": True})
    ev = evaluate_kpi(cfg, load_dataset(cfg), kpi, preds, None)
    # preditor aleatório proporcional às classes: F1_l = p_l, média sobre K classes presentes = 1/K
    assert ev.chance == pytest.approx(1 / k)
