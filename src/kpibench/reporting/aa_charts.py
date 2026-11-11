"""Gráficos executivos do teste de ligações (matplotlib, sem plotly).

Linguagem visual inspirada em painéis de benchmark de LLMs (ex.: Artificial Analysis):
uma pergunta por gráfico, título que afirma a métrica e o sentido ("maior é melhor"),
barras ordenadas com rótulo de valor, cor fixa por PROVEDOR, quadrante "mais atrativo"
no gráfico de qualidade × custo e rodapé com fonte, amostra e referência.

Cores: slots 1–6 da paleta categórica validada (adjacentes passam CVD em modo claro).
Três slots ficam abaixo de 3:1 de contraste, por isso TODA barra e todo ponto têm
rótulo direto. Texto sempre em tinta neutra; a cor só identifica o provedor.
"""
from __future__ import annotations

import textwrap

import matplotlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FuncFormatter

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
GOOD = "#006300"
CRITICAL = "#d03b3b"
NEUTRAL = "#a9a8a2"           # produção atual

PROVIDER_COLORS = {            # ordem fixa = slots 1..7 da paleta (nunca reciclar)
    "OpenAI": "#2a78d6",
    "Anthropic": "#eb6834",
    "Google": "#1baf7a",
    "xAI": "#eda100",
    "DeepSeek": "#e87ba4",
    "Moonshot": "#008300",
    "Outros": "#6250d6",
    "Produção atual": NEUTRAL,
}
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#104281"]


def setup_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 10, "axes.titlesize": 14, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.titlepad": 26, "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.labelsize": 10,
        "xtick.color": INK_2, "ytick.color": INK_2, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.axisbelow": True, "legend.frameon": False, "legend.fontsize": 9,
        "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
    })


setup_style()


# =============================================================================== helpers
def br(v: float, nd: int = 1) -> str:
    """Número no formato brasileiro (vírgula decimal, ponto de milhar)."""
    if v is None or not np.isfinite(v):
        return "—"
    s = f"{v:,.{nd}f}"
    return s.replace(",", "§").replace(".", ",").replace("§", ".")


def usd(v: float) -> str:
    if v is None or not np.isfinite(v):
        return "—"
    return f"US$ {br(v, 2)}" if v >= 0.1 else f"US$ {br(v, 3)}"


def _br_axis(nd: int = 1, prefix: str = "", suffix: str = ""):
    return FuncFormatter(lambda x, _: f"{prefix}{br(x, nd)}{suffix}")


def _color(provider: str) -> str:
    return PROVIDER_COLORS.get(provider, PROVIDER_COLORS["Outros"])


def _header(ax, title: str, subtitle: str) -> None:
    """Título + subtítulo com distâncias fixas em pontos (não dependem da altura do gráfico)."""
    ax.set_title(title, color=INK, pad=28)
    ax.annotate(subtitle, xy=(0, 1), xycoords="axes fraction", xytext=(0, 9), textcoords="offset points",
                color=INK_2, fontsize=10, va="bottom", ha="left")


def _footer(fig, footer: str) -> None:
    if footer:
        fig.text(0.01, 0.005, footer, color=MUTED, fontsize=8, ha="left", va="bottom")


def _provider_legend(ax, providers, extra: list | None = None, loc="upper right", ncol=None,
                     inside: bool = False) -> None:
    """Legenda por provedor. inside=True: dentro da área do gráfico (não disputa espaço com o subtítulo)."""
    handles = [Patch(facecolor=_color(p), label=p) for p in providers]
    handles += extra or []
    kw = dict(handlelength=1.0, handleheight=1.0, columnspacing=1.2, ncol=ncol or min(len(handles), 4))
    if inside:
        leg = ax.legend(handles=handles, loc=loc, frameon=True, **kw)
        leg.get_frame().set_facecolor(SURFACE)
        leg.get_frame().set_edgecolor(GRID)
    else:
        ax.legend(handles=handles, loc=loc, bbox_to_anchor=(1, 1.13), **kw)


def _ordered_providers(series: pd.Series) -> list[str]:
    order = list(PROVIDER_COLORS)
    return sorted(series.unique(), key=lambda p: order.index(p) if p in order else 99)


# =============================================================================== scorecards
def scorecards(board: pd.DataFrame, routing: pd.DataFrame, n_calls: int, footer: str = ""):
    """Quatro números que resumem a decisão (stat tiles)."""
    best = board.iloc[0]
    value = board.sort_values("indice_por_dolar", ascending=False).iloc[0]
    fastest = board.sort_values("lat_p50").iloc[0]
    routed = float(routing["usd_1k_recomendado"].sum()) if "usd_1k_recomendado" in routing else np.nan
    tiles = [
        ("Maior concordância", br(best["indice"], 1), "de 100", best["model"]),
        ("Melhor índice por dólar", br(value["indice"], 1), f"por {usd(value['usd_1k_ligacoes'])} / 1.000 lig.",
         value["model"]),
        ("Roteamento por KPI", usd(routed), "por 1.000 ligações", f"{routing['recomendado'].nunique()} modelo(s)"),
        ("Menor latência (p50)", f"{br(fastest['lat_p50'], 1)} s", "por chamada de KPI", fastest["model"]),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(14, 2.3))
    for ax, (label, big, unit, who) in zip(axes, tiles):
        ax.axis("off")
        ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor="white", edgecolor=GRID,
                               linewidth=1.0))
        ax.text(0.07, 0.80, label.upper(), transform=ax.transAxes, fontsize=8.5, color=INK_2, weight="bold")
        ax.text(0.07, 0.43, big, transform=ax.transAxes, fontsize=24, color=INK, weight="bold")
        ax.text(0.07, 0.25, unit, transform=ax.transAxes, fontsize=9, color=INK_2)
        ax.text(0.07, 0.08, who, transform=ax.transAxes, fontsize=9.5, color=INK)
    fig.subplots_adjust(wspace=0.08, left=0.01, right=0.99, top=0.95, bottom=0.12)
    _footer(fig, footer or f"{n_calls} ligações avaliadas")
    return fig


# =============================================================================== volume
def calls_per_day(calls: pd.DataFrame, footer: str = "", date_col: str = "data_referencia"):
    if date_col not in calls or calls[date_col].isna().all():
        return None
    d = pd.to_datetime(calls[date_col]).dt.date.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10, 3.4))
    x = np.arange(len(d))
    ax.bar(x, d.values, color=SEQ_BLUE[4], width=0.7, edgecolor=SURFACE, linewidth=1.2)
    for xi, v in zip(x, d.values):
        ax.text(xi, v, f"{v}", ha="center", va="bottom", fontsize=8, color=INK)
    step = max(1, len(d) // 12)
    ax.set_xticks(x[::step], [pd.Timestamp(i).strftime("%d/%m") for i in d.index][::step])
    ax.set_ylabel("Ligações")
    _header(ax, "Ligações testadas por dia", f"{len(calls)} ligações entre {pd.Timestamp(d.index.min()):%d/%m/%Y} "
            f"e {pd.Timestamp(d.index.max()):%d/%m/%Y}")
    _footer(fig, footer)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


# =============================================================================== leaderboard
def leaderboard(board: pd.DataFrame, footer: str = "", reference: str = ""):
    """Índice de concordância (0–100) por modelo, barras ordenadas, cor por provedor."""
    b = board.sort_values("indice", ascending=False)
    fig, ax = plt.subplots(figsize=(max(8, 0.75 * len(b) + 3), 5.2))
    x = np.arange(len(b))
    ax.bar(x, b["indice"], color=[_color(p) for p in b["provedor"]], width=0.68,
           edgecolor=SURFACE, linewidth=1.5, zorder=2)
    for xi, v in zip(x, b["indice"]):
        ax.text(xi, v + 1.2, br(v, 1), ha="center", va="bottom", fontsize=9.5, color=INK, weight="bold")
    ax.set_xticks(x, b["model"], rotation=40, ha="right")
    ax.set_ylim(0, 122)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("Índice de concordância (0–100)")
    ax.yaxis.set_major_formatter(_br_axis(0))
    _header(ax, "Índice de concordância kpibench",
            f"Média da métrica primária por KPI (QWK, macro-F1, nota do juiz) vs. {reference or 'referência'}"
            " · maior é melhor")
    _provider_legend(ax, _ordered_providers(b["provedor"]), inside=True, ncol=len(b["provedor"].unique()))
    _footer(fig, footer)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


# =============================================================================== qualidade × custo
def quality_vs_cost(board: pd.DataFrame, footer: str = ""):
    """Concordância × custo por 1.000 ligações (log), com o quadrante mais atrativo destacado."""
    b = board.dropna(subset=["usd_1k_ligacoes"])
    fig, ax = plt.subplots(figsize=(10, 6.2))
    xmed, ymed = float(np.median(b["usd_1k_ligacoes"])), float(np.median(b["indice"]))
    xmin, xmax = b["usd_1k_ligacoes"].min() / 1.8, b["usd_1k_ligacoes"].max() * 1.8
    ymin = max(0, b["indice"].min() - 8)
    ymax = min(105, b["indice"].max() + 6)
    ax.add_patch(Rectangle((xmin, ymed), xmed - xmin, ymax - ymed, facecolor="#e6f2e6", edgecolor="none", zorder=0))
    ax.text(xmin * 1.08, ymax - 0.8, "Quadrante mais atrativo\n(acima da mediana, abaixo do custo mediano)",
            fontsize=8.5, color=GOOD, va="top", weight="bold")
    ax.axvline(xmed, color=AXIS, lw=0.8, ls="--", zorder=1)
    ax.axhline(ymed, color=AXIS, lw=0.8, ls="--", zorder=1)
    for r in b.itertuples():
        c = _color(r.provedor)
        ax.scatter(r.usd_1k_ligacoes, r.indice, s=110, color=c if not r.preco_estimado else SURFACE,
                   edgecolor=c, linewidth=2.2, zorder=3)
        ax.annotate(r.model, (r.usd_1k_ligacoes, r.indice), xytext=(7, 5), textcoords="offset points",
                    fontsize=8.5, color=INK)
    ax.set_xscale("log")
    from matplotlib.ticker import LogLocator
    ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=()))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: usd(v)))
    ax.yaxis.set_major_formatter(_br_axis(0))
    ax.set_xlabel("Custo para extrair os KPIs selecionados de 1.000 ligações (escala log)")
    ax.set_ylabel("Índice de concordância (0–100)")
    _header(ax, "Concordância × custo", "Canto superior esquerdo = mais qualidade por menos dinheiro")
    extra = [Line2D([], [], marker="o", ls="", markerfacecolor=SURFACE, markeredgecolor=INK_2, markersize=8,
                    markeredgewidth=2, label="preço estimado")]
    _provider_legend(ax, _ordered_providers(b["provedor"]), extra=extra)
    _footer(fig, footer)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


# =============================================================================== custo
def cost_bars(board: pd.DataFrame, footer: str = ""):
    b = board.sort_values("usd_1k_ligacoes")
    fig, ax = plt.subplots(figsize=(max(8, 0.75 * len(b) + 3), 5.0))
    x = np.arange(len(b))
    for xi, r in zip(x, b.itertuples()):
        c = _color(r.provedor)
        ax.bar(xi, r.usd_1k_ligacoes, color=c, width=0.68, edgecolor=SURFACE if not r.preco_estimado else INK_2,
               linewidth=1.5 if not r.preco_estimado else 0.8, hatch="///" if r.preco_estimado else None, zorder=2)
        ax.text(xi, r.usd_1k_ligacoes * 1.02, usd(r.usd_1k_ligacoes), ha="center", va="bottom", fontsize=8.5,
                color=INK, weight="bold")
    ax.set_xticks(x, b["model"], rotation=40, ha="right")
    ax.set_ylabel("US$ por 1.000 ligações")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"US$ {br(v, 0)}"))
    ax.set_ylim(0, b["usd_1k_ligacoes"].max() * 1.22)
    _header(ax, "Custo por 1.000 ligações",
            "Tokens medidos nesta rodada × preço de config/models.yaml (todos os KPIs selecionados) · menor é melhor")
    extra = [Patch(facecolor="white", edgecolor=INK_2, hatch="///", label="preço estimado")]
    _provider_legend(ax, _ordered_providers(b["provedor"]), extra=extra, loc="upper left", inside=True, ncol=2)
    _footer(fig, footer)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


# =============================================================================== latência
def latency_chart(board: pd.DataFrame, footer: str = "", sla_s: float | None = None):
    b = board.sort_values("lat_p50", ascending=False)
    fig, ax = plt.subplots(figsize=(10, max(3.8, 0.42 * len(b) + 1.8)))
    y = np.arange(len(b))
    for yi, r in zip(y, b.itertuples()):
        c = _color(r.provedor)
        ax.plot([r.lat_p50, r.lat_p95], [yi, yi], color=c, lw=6, alpha=0.35, solid_capstyle="round", zorder=2)
        ax.scatter(r.lat_p50, yi, s=70, color=c, zorder=3, edgecolor=SURFACE, linewidth=1.5)
        ax.scatter(r.lat_p95, yi, s=28, color=SURFACE, edgecolor=c, linewidth=1.8, zorder=3)
        ax.text(r.lat_p95 + 0.12, yi, f"{br(r.lat_p50, 1)} s  ·  p95 {br(r.lat_p95, 1)} s", va="center",
                fontsize=8.5, color=INK)
    if sla_s:
        ax.axvline(sla_s, color=CRITICAL, lw=1.2, ls="--", zorder=1)
        ax.text(sla_s, -0.75, f" SLA {br(sla_s, 1)} s", color=CRITICAL, fontsize=8.5, va="center",
                weight="bold")
    ax.set_yticks(y, b["model"])
    ax.set_ylim(-1.1, len(b) - 0.5)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_xlim(0, max(b["lat_p95"].max(), sla_s or 0) * 1.35)
    ax.xaxis.set_major_formatter(_br_axis(0, suffix=" s"))
    ax.set_xlabel("Segundos por chamada de KPI (ponto cheio = mediana · ponto vazado = p95)")
    _header(ax, "Latência por chamada", "Tempo até a resposta completa no gateway Nexus · menor é melhor")
    _footer(fig, footer)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


# =============================================================================== confiabilidade
def reliability_bars(board: pd.DataFrame, footer: str = ""):
    """Falhas (JSON inválido ou valor fora da escala/taxonomia) por 1.000 chamadas — eixo a partir de zero."""
    b = board.assign(falhas=(1 - board["taxa_resposta_valida"]) * 1000).sort_values("falhas", ascending=False)
    fig, ax = plt.subplots(figsize=(10, max(3.6, 0.38 * len(b) + 1.6)))
    y = np.arange(len(b))
    ax.barh(y, b["falhas"], color=[_color(p) for p in b["provedor"]], height=0.62, edgecolor=SURFACE, linewidth=1.5)
    top = max(b["falhas"].max(), 1.0)
    for yi, v in zip(y, b["falhas"]):
        ax.text(v + top * 0.015, yi, br(v, 1) if v > 0 else "0 (nenhuma falha)", va="center", fontsize=8.5,
                color=INK)
    ax.set_yticks(y, b["model"])
    ax.set_xlim(0, top * 1.25)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.xaxis.set_major_formatter(_br_axis(0))
    ax.set_xlabel("Respostas inválidas por 1.000 chamadas de KPI")
    _header(ax, "Confiabilidade da resposta",
            "JSON quebrado ou valor fora da escala/taxonomia conta como erro na avaliação · menor é melhor")
    _footer(fig, footer)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


# =============================================================================== heatmap
def kpi_heatmap(metrics: pd.DataFrame, routing: pd.DataFrame, footer: str = ""):
    m = metrics[metrics["model"] != "producao_atual"]
    disp = dict(zip(routing["kpi"], routing["display"])) if "display" in routing else {}
    piv = m.pivot_table(index="kpi", columns="model", values="primary")
    piv = piv[piv.mean().sort_values(ascending=False).index]
    piv = piv.loc[[k for k in routing["kpi"] if k in piv.index]] if len(routing) else piv
    rec = dict(zip(routing["kpi"], routing["recomendado"])) if "recomendado" in routing else {}
    cmap = LinearSegmentedColormap.from_list("seq", SEQ_BLUE)
    vals = piv.to_numpy(float)
    fig, ax = plt.subplots(figsize=(max(8, 0.9 * piv.shape[1] + 4), 0.55 * piv.shape[0] + 2.4))
    vmin, vmax = np.nanmin(vals), np.nanmax(vals)
    ax.imshow(vals, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    norm = (vals - vmin) / max(vmax - vmin, 1e-9)
    for i, k in enumerate(piv.index):
        for j, mdl in enumerate(piv.columns):
            if np.isfinite(vals[i, j]):
                ax.text(j, i, br(vals[i, j], 2), ha="center", va="center", fontsize=8.5,
                        color="white" if norm[i, j] > 0.55 else INK, weight="bold" if rec.get(k) == mdl else None)
                if rec.get(k) == mdl:
                    ax.add_patch(Rectangle((j - 0.48, i - 0.46), 0.96, 0.92, fill=False, edgecolor=INK, lw=1.8))
    ax.set_xticks(range(piv.shape[1]), piv.columns, rotation=40, ha="right")
    ax.set_yticks(range(piv.shape[0]), [disp.get(k, k) for k in piv.index])
    ax.tick_params(length=0)
    _header(ax, "Concordância por KPI e modelo",
            "Métrica primária de cada KPI · contorno = modelo recomendado (mais barato não-inferior ao melhor)")
    _footer(fig, footer)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


# =============================================================================== small multiples
def kpi_small_multiples(metrics: pd.DataFrame, routing: pd.DataFrame, footer: str = "", ncols: int = 2):
    from kpibench.live import provider_of

    kpis = list(routing["kpi"]) if len(routing) else list(metrics["kpi"].unique())
    disp = dict(zip(routing["kpi"], routing["display"])) if "display" in routing else {}
    rec = dict(zip(routing["kpi"], routing["recomendado"])) if "recomendado" in routing else {}
    nrows = int(np.ceil(len(kpis) / ncols))
    n_models = metrics["model"].nunique()
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * (0.32 * n_models + 1.4)), squeeze=False)
    for ax, kpi in zip(axes.flat, kpis):
        g = metrics[metrics["kpi"] == kpi].sort_values("primary")
        y = np.arange(len(g))
        for yi, r in zip(y, g.itertuples()):
            prov = provider_of(r.model)
            col = _color(prov)
            inferior = r.decision == "inferior"
            if np.isfinite(r.ci_low) and np.isfinite(r.ci_high):
                ax.plot([r.ci_low, r.ci_high], [yi, yi], color=col, lw=3, alpha=0.30 if inferior else 0.45,
                        solid_capstyle="round", zorder=2)
            ax.scatter(r.primary, yi, s=64, color=SURFACE if inferior else col, edgecolor=col, linewidth=2,
                       zorder=3)
            tag = "  ★ recomendado" if rec.get(kpi) == r.model else ""
            right = max(r.ci_high if np.isfinite(r.ci_high) else r.primary, r.primary)
            ax.text(right + 0.012, yi, f"{br(r.primary, 2)}{tag}", va="center", fontsize=8, color=INK,
                    weight="bold" if tag else None)
        ax.set_yticks(y, g["model"], fontsize=8)
        ax.set_ylim(-0.7, len(g) - 0.3)
        lo = max(0.0, np.floor(((g["ci_low"].min() if g["ci_low"].notna().any() else g["primary"].min()) - 0.03)
                               * 10) / 10)
        ax.set_xlim(lo, 1.17)
        ax.set_xticks(np.round(np.arange(lo, 1.0001, 0.1), 1))
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.xaxis.set_major_formatter(_br_axis(1))
        ax.set_title(f"{disp.get(kpi, kpi)}  ·  {g['primary_metric'].iloc[0]}", fontsize=10.5, pad=6)
    for ax in list(axes.flat)[len(kpis):]:
        ax.set_visible(False)
    fig.suptitle("Desempenho por KPI", x=0.01, ha="left", fontsize=14, weight="bold", color=INK, y=1.0)
    fig.text(0.01, 0.975, "Ponto = estimativa · faixa = IC 95% (bootstrap pareado) · ponto vazado = inferior ao melhor "
             "(IC acima da margem) · cinza = produção atual · eixo não começa em zero", fontsize=9.5, color=INK_2, ha="left", va="top")
    _footer(fig, footer)
    fig.tight_layout(rect=(0, 0.02, 1, 0.965))
    return fig


# =============================================================================== produção × recomendado
def production_dumbbell(cmp: pd.DataFrame, footer: str = ""):
    """KPIs comparáveis: de onde a produção está para onde o modelo recomendado leva."""
    from kpibench.live import provider_of

    if cmp.empty:
        return None
    c = cmp.sort_values("ganho_recomendado_pp")
    fig, ax = plt.subplots(figsize=(10, 0.75 * len(c) + 2.2))
    y = np.arange(len(c))
    for yi, r in zip(y, c.itertuples()):
        col = _color(provider_of(r.recomendado))
        ax.plot([r.producao, r.perf_recomendado], [yi, yi], color=GRID, lw=4, zorder=1, solid_capstyle="round")
        ax.scatter(r.producao, yi, s=90, color=NEUTRAL, zorder=2, edgecolor=SURFACE, lw=1.5)
        ax.scatter(r.perf_recomendado, yi, s=110, color=col, zorder=3, edgecolor=SURFACE, lw=1.5)
        sign = "+" if r.ganho_recomendado_pp >= 0 else "−"
        ax.text(max(r.producao, r.perf_recomendado) + 0.025, yi,
                f"{sign}{br(abs(r.ganho_recomendado_pp), 1)} p.p.  ({r.recomendado})", va="center", fontsize=8.5,
                color=GOOD if r.ganho_recomendado_pp >= 0 else CRITICAL, weight="bold")
        ax.text(min(r.producao, r.perf_recomendado) - 0.02, yi, br(min(r.producao, r.perf_recomendado), 2),
                va="center", ha="right", fontsize=8, color=INK_2)
    ax.set_yticks(y, [f"{d}  ·  {m}" for d, m in zip(c["display"], c["metrica"])])
    lo = max(0.0, np.floor((min(c["producao"].min(), c["perf_recomendado"].min()) - 0.1) * 10) / 10)
    ax.set_xlim(lo, 1.32)
    ax.set_xticks(np.round(np.arange(lo, 1.0001, 0.1), 1))
    ax.set_ylim(-0.8, len(c) - 0.4)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.xaxis.set_major_formatter(_br_axis(1))
    _header(ax, "Produção atual × modelo recomendado",
            "Concordância com a mesma referência, só nos KPIs em que a produção tem domínio comparável")
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=NEUTRAL, markersize=9, label="produção atual"),
                       Line2D([], [], marker="o", ls="", color=INK_2, markersize=9,
                              label="recomendado (cor do provedor)")],
              loc="lower right", ncol=2, frameon=True, facecolor=SURFACE, edgecolor=GRID)
    _footer(fig, footer)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


# =============================================================================== tabelas
# Observação: o renderer de notebook (Jupyter/VS Code) aplica, por padrão, um zebra-striping
# (fundo cinza leve em linhas ímpares via CSS da classe "dataframe"/"rendered_html") que não vem
# do código abaixo. Como o Styler do pandas prefixa cada seletor com o id único da tabela
# (`#T_xxxxx ...`), essas regras têm mais especificidade do que a CSS padrão do notebook e
# sobrescrevem qualquer sombreamento herdado — todas as linhas e células ficam com fundo plano.
_TABLE_STYLES = [
    {"selector": "table", "props": [("border-collapse", "collapse"), ("box-shadow", "none")]},
    {"selector": "th", "props": [("background-color", "#f3f2ee"), ("color", INK), ("font-weight", "600"),
                                 ("text-align", "left"), ("padding", "6px 10px"), ("vertical-align", "bottom"),
                                 ("box-shadow", "none")]},
    {"selector": "td", "props": [("padding", "6px 10px"), ("color", INK), ("background-color", SURFACE),
                                 ("box-shadow", "none")]},
    {"selector": "tbody tr", "props": [("background-color", SURFACE), ("box-shadow", "none")]},
    {"selector": "tbody tr:nth-child(odd)", "props": [("background-color", SURFACE)]},
    {"selector": "tbody tr:nth-child(even)", "props": [("background-color", SURFACE)]},
    {"selector": "tbody tr:hover", "props": [("background-color", SURFACE)]},
    {"selector": "caption", "props": [("caption-side", "top")]},
]


def _ci_br(s: str) -> str:
    try:
        lo, hi = (float(x) for x in str(s).strip("[]").split(","))
        return f"[{br(lo, 3)}; {br(hi, 3)}]"
    except ValueError:
        return s


def style_routing(routing: pd.DataFrame, board: pd.DataFrame | None = None):
    cols = {"display": "KPI", "metrica": "Métrica", "recomendado": "Recomendado", "perf_recomendado": "Concordância",
            "ci_recomendado": "IC 95%", "usd_1k_recomendado": "US$ / 1.000 lig.", "melhor": "Melhor",
            "perf_melhor": "Concordância do melhor", "economia_vs_melhor_pct": "Economia vs melhor",
            "n_inconclusivos": "Comparações inconclusivas"}
    t = routing[[c for c in cols if c in routing.columns]].rename(columns=cols)
    est = set(board.loc[board["preco_estimado"], "model"]) if board is not None else set()
    t["Recomendado"] = t["Recomendado"].map(lambda m: f"{m} *" if m in est else m)
    if "IC 95%" in t:
        t["IC 95%"] = t["IC 95%"].map(_ci_br)
    fmt = {"Concordância": lambda v: br(v, 3), "Concordância do melhor": lambda v: br(v, 3),
           "US$ / 1.000 lig.": usd, "Economia vs melhor": lambda v: f"{br(v, 0)}%"}
    text_cols = [c for c in ("KPI", "Métrica", "Recomendado", "IC 95%", "Melhor") if c in t]
    sty = (t.style.format({k: v for k, v in fmt.items() if k in t.columns}).hide(axis="index")
           .set_table_styles(_TABLE_STYLES)
           .set_properties(subset=text_cols, **{"text-align": "left"}))
    return sty


def style_scoreboard(board: pd.DataFrame):
    cols = {"model": "Modelo", "provedor": "Provedor", "indice": "Índice (0–100)",
            "kpis_nao_inferiores": "KPIs não-inferiores", "usd_1k_ligacoes": "US$ / 1.000 lig.",
            "indice_por_dolar": "Índice por US$", "lat_p50": "Latência p50", "lat_p95": "Latência p95",
            "taxa_resposta_valida": "Respostas válidas", "preco_estimado": "Preço estimado"}
    t = board[[c for c in cols if c in board.columns]].rename(columns=cols).copy()
    if "KPIs não-inferiores" in t and "n_kpis" in board:
        t["KPIs não-inferiores"] = [f"{a} de {b}" for a, b in zip(board["kpis_nao_inferiores"], board["n_kpis"])]
    t["Preço estimado"] = t["Preço estimado"].map({True: "sim", False: "não"})
    fmt = {"Índice (0–100)": lambda v: br(v, 1), "US$ / 1.000 lig.": usd, "Índice por US$": lambda v: br(v, 1),
           "Latência p50": lambda v: f"{br(v, 1)} s", "Latência p95": lambda v: f"{br(v, 1)} s",
           "Respostas válidas": lambda v: f"{br(100 * v, 1)}%"}
    return (t.style.format(fmt).hide(axis="index").set_table_styles(_TABLE_STYLES)
            .set_properties(subset=["Modelo", "Provedor", "KPIs não-inferiores", "Preço estimado"],
                            **{"text-align": "left"}))


def style_drilldown(df: pd.DataFrame, resumo: str | None = None):
    text_rows = set(df.attrs.get("text_rows", []))

    def mark(row):
        ref = row.get("Referência")
        out = []
        for c, v in row.items():
            if c == "Referência" or row.name in text_rows or pd.isna(v) or pd.isna(ref):
                out.append("")
            else:
                # sinal leve por cor do texto (sem preenchimento de fundo): verde = concorda com a
                # referência, vermelho em negrito = diverge.
                out.append("color:#2e7d32" if str(v) == str(ref) else "color:#c62828;font-weight:600")
        return out

    def short(v):
        return "\n".join(textwrap.wrap(str(v), 48)) if isinstance(v, str) and len(v) > 48 else v

    sty = (df.map(short).style.apply(mark, axis=1).format(na_rep="—")
           .set_properties(**{"white-space": "pre-wrap", "font-size": "12px", "vertical-align": "top",
                              "text-align": "left"})
           .set_table_styles(_TABLE_STYLES))
    if resumo:
        sty = sty.set_caption(f"<div style='text-align:left;max-width:1100px;color:{INK_2};font-size:12px;"
                              f"padding-bottom:6px'><b>Resumo:</b> {resumo}</div>")
    return sty
