"""Figuras estáticas (PNG) do relatório.

Paleta: slots categóricos 1-2 da paleta de referência validada (azul/laranja);
sequencial azul de um só matiz; texto sempre em tinta neutra, nunca na cor da série.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e6e5e1"
MUTED = "#a9a8a2"
S1 = "#2a78d6"   # azul
S2 = "#eb6834"   # laranja

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
    "axes.titlesize": 10, "axes.titleweight": "bold",
})


def pareto_grid(full: pd.DataFrame, rec: pd.DataFrame, out: Path) -> Path:
    kpis = list(rec["kpi"])
    ncols = 2
    nrows = math.ceil(len(kpis) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 3.4 * nrows), squeeze=False)
    for ax, kpi in zip(axes.flat, kpis):
        t = full[(full["kpi"] == kpi) & full["proj_usd_per_1k"].notna()]
        r = rec[rec["kpi"] == kpi].iloc[0]
        for row in t.itertuples():
            on_front = bool(row.pareto)
            color = S1 if on_front else MUTED
            ax.errorbar(row.proj_usd_per_1k, row.primary,
                        yerr=[[row.primary - row.ci_low], [row.ci_high - row.primary]],
                        fmt="o", ms=7, color=color, ecolor=color, elinewidth=1.2, capsize=0, zorder=3,
                        mec=SURFACE, mew=1.5)
            if row.model == r["recomendado"]:
                ax.scatter([row.proj_usd_per_1k], [row.primary], s=180, facecolors="none",
                           edgecolors=INK, linewidths=1.4, zorder=4)
            ax.annotate(row.model, (row.proj_usd_per_1k, row.primary), xytext=(6, 4),
                        textcoords="offset points", fontsize=7.5, color=INK_2)
        f = t[t["pareto"]].sort_values("proj_usd_per_1k")
        ax.plot(f["proj_usd_per_1k"], f["primary"], color=S1, lw=1.5, alpha=0.6, zorder=2)
        ax.set_xscale("log")
        ax.set_title(f"{r['display']}  ·  {r['metrica']}", loc="left")
        ax.set_xlabel("USD por 1.000 chamadas (projetado, log)")
        ax.set_ylabel(r["metrica"])
    for ax in list(axes.flat)[len(kpis):]:
        ax.set_visible(False)
    fig.suptitle("Custo x performance por KPI — azul: fronteira de Pareto · anel: recomendado · barras: IC 95%",
                 x=0.01, ha="left", fontsize=10, color=INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = out / "pareto_por_kpi.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def heatmap(full: pd.DataFrame, rec: pd.DataFrame, out: Path) -> Path:
    piv = full.pivot_table(index="kpi", columns="model", values="primary").reindex(rec["kpi"])
    piv = piv[piv.mean().sort_values(ascending=False).index]
    cmap = LinearSegmentedColormap.from_list("seq_blue", ["#eef4fc", "#9cc1ee", "#2a78d6", "#0d3d7a"])
    fig, ax = plt.subplots(figsize=(1.1 * piv.shape[1] + 3, 0.5 * piv.shape[0] + 1.8))
    vals = piv.to_numpy(float)
    im = ax.imshow(vals, cmap=cmap, vmin=np.nanmin(vals), vmax=np.nanmax(vals), aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(piv.shape[1]), piv.columns, rotation=30, ha="right")
    ax.set_yticks(range(piv.shape[0]), piv.index)
    norm = (vals - np.nanmin(vals)) / max(np.nanmax(vals) - np.nanmin(vals), 1e-9)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            if np.isfinite(vals[i, j]):
                ax.text(j, i, f"{vals[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                        color="#ffffff" if norm[i, j] > 0.55 else INK)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.outline.set_visible(False)
    ax.set_title("Métrica primária por KPI e modelo (QWK · macro-F1 · nota do juiz)", loc="left")
    fig.tight_layout()
    path = out / "heatmap_kpi_modelo.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def summary_meta_eval(meta: pd.DataFrame, out: Path) -> Path | None:
    m = meta[meta["status"].eq("ok") & meta.get("spearman", pd.Series(dtype=float)).notna()] \
        if "spearman" in meta.columns else pd.DataFrame()
    if m.empty:
        return None
    m = m.sort_values("spearman")
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(m) + 1.6))
    colors = [S1 if g == "eficiente" else S2 for g in m["grupo"]]
    y = np.arange(len(m))
    lo = (m["spearman"] - m["spearman_ci_low"]).clip(lower=0)
    hi = (m["spearman_ci_high"] - m["spearman"]).clip(lower=0)
    ax.barh(y, m["spearman"], color=colors, height=0.62, zorder=2)
    ax.errorbar(m["spearman"], y, xerr=[lo, hi], fmt="none", ecolor=INK_2, elinewidth=1, zorder=3)
    ax.set_yticks(y, m["metric"])
    ax.axvline(0, color=INK_2, lw=0.8)
    ax.set_xlabel("Spearman com a nota humana de qualidade (IC 95%)")
    ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch

    ax.legend(handles=[Patch(color=S1, label="eficiente (CPU, ~US$0)"),
                       Patch(color=S2, label="moderna (modelo local ou LLM)")],
              loc="lower right", frameon=False)
    ax.set_title("Qualidade do resumo: concordância de cada métrica com a avaliação humana", loc="left")
    fig.tight_layout()
    path = out / "meta_avaliacao_metricas_resumo.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
