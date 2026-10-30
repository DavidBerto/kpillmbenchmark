"""Métricas para KPIs ordinais e categóricos."""
from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score, f1_score


def qwk(y_true, y_pred, lo: int, hi: int) -> float:
    """Quadratic Weighted Kappa: concordância ordinal corrigida pelo acaso.
    Penaliza erro de 2 pontos 4x mais que erro de 1 ponto.
    Implementação vetorizada (equivalente a sklearn cohen_kappa_score(weights='quadratic'))."""
    y_true, y_pred = np.asarray(y_true, int) - lo, np.asarray(y_pred, int) - lo
    n, k = len(y_true), hi - lo + 1
    if n < 2:
        return float("nan")
    obs = np.bincount(y_true * k + y_pred, minlength=k * k).reshape(k, k).astype(float)
    exp = np.outer(obs.sum(1), obs.sum(0)) / n
    i = np.arange(k)
    w = (i[:, None] - i[None, :]) ** 2
    den = (w * exp).sum()
    return float(1 - (w * obs).sum() / den) if den > 0 else float("nan")


def spearman(y_true, y_pred) -> float:
    if len(set(np.asarray(y_true).tolist())) < 2 or len(set(np.asarray(y_pred).tolist())) < 2:
        return float("nan")
    return float(spearmanr(y_true, y_pred).statistic)


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float)))) if len(y_true) else float("nan")


def exact_acc(y_true, y_pred) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred))) if len(y_true) else float("nan")


def within1_acc(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float)) <= 1)) if len(y_true) else float("nan")


def macro_f1(y_true, y_pred, labels: list[str]) -> float:
    """Macro-F1 sobre os rótulos da taxonomia. Predições inválidas não pertencem a
    nenhum rótulo: derrubam o recall da classe verdadeira (contam como erro)."""
    if not len(y_true):
        return float("nan")
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    f1s = []
    for l in labels:  # só classes com suporte (evita 0s artificiais de classes ausentes)
        t = y_true == l
        if not t.any():
            continue
        p = y_pred == l
        tp = float(np.sum(t & p))
        denom = t.sum() + p.sum()
        f1s.append(2 * tp / denom if denom else 0.0)
    return float(np.mean(f1s)) if f1s else float("nan")


def cohen_kappa(y_true, y_pred) -> float:
    if len(y_true) < 2:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        k = cohen_kappa_score(y_true, y_pred)
    return float(k) if np.isfinite(k) else float("nan")


def per_class_f1(y_true, y_pred, labels: list[str]) -> dict[str, float]:
    scores = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    support = {l: int(np.sum(np.asarray(y_true) == l)) for l in labels}
    return {l: (float(s), support[l]) for l, s in zip(labels, scores)}
