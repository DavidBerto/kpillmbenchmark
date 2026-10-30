"""Bootstrap não-paramétrico e comparação pareada entre modelos.

Todos os modelos são reamostrados com os MESMOS índices de registro (bootstrap
pareado), o que elimina a variância entre itens da diferença entre modelos e
dá ICs bem mais estreitos para "A é melhor que B?" do que comparar ICs isolados.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


def bootstrap_indices(n: int, n_boot: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(n_boot, n))


def bootstrap_dist(fn: Callable[[np.ndarray], float], idx: np.ndarray) -> np.ndarray:
    return np.array([fn(row) for row in idx], dtype=float)


def percentile_ci(dist: np.ndarray, ci: float) -> tuple[float, float]:
    d = dist[np.isfinite(dist)]
    if not len(d):
        return float("nan"), float("nan")
    a = (1 - ci) / 2
    return float(np.quantile(d, a)), float(np.quantile(d, 1 - a))


@dataclass
class PairedComparison:
    model: str
    reference: str
    diff: float            # referência - modelo (positivo = modelo pior)
    ci_low: float
    ci_high: float
    p_worse: float         # P_bootstrap(modelo < referência)
    decision: str          # referencia | nao_inferior | inferior | inconclusivo


def compare_to_reference(point: dict[str, float], dists: dict[str, np.ndarray], reference: str,
                         margin: float, ci: float) -> list[PairedComparison]:
    """Teste de não-inferioridade via IC bootstrap pareado da diferença.

    nao_inferior : limite superior do IC da diferença < margem
    inferior     : limite inferior do IC da diferença > margem (pior com confiança)
    inconclusivo : caso contrário -> amostra insuficiente para decidir
    """
    out = []
    ref_d = dists[reference]
    for m, d in dists.items():
        if m == reference:
            out.append(PairedComparison(m, reference, 0.0, 0.0, 0.0, 0.0, "referencia"))
            continue
        diff = ref_d - d
        lo, hi = percentile_ci(diff, ci)
        p_worse = float(np.mean(diff[np.isfinite(diff)] > 0)) if np.isfinite(diff).any() else float("nan")
        if np.isfinite(hi) and hi < margin:
            dec = "nao_inferior"
        elif np.isfinite(lo) and lo > margin:
            dec = "inferior"
        else:
            dec = "inconclusivo"
        out.append(PairedComparison(m, reference, point[reference] - point[m], lo, hi, p_worse, dec))
    return out
