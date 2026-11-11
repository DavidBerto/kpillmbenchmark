"""Gráficos da comparação de backends (notebook 05): LLM × CDM × Jev × híbrido.

Mesma linguagem de aa_charts: uma pergunta por gráfico, título que afirma a métrica, rótulo direto.
Cor fixa por TIPO de backend (atual = cinza; consolidado = azul; CDM = verde; Jev = laranja; híbrido = roxo).
"""
from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

from kpibench.reporting.aa_charts import (AXIS, CRITICAL, GOOD, GRID, INK, INK_2, MUTED, SURFACE, _br_axis,
                                          _footer, _header, br, usd)

KIND_COLORS = {"llm_per_kpi": "#6f6e69", "llm_single_call": "#2a78d6", "llm_json_schema": "#e87ba4",
               "cdm": "#1baf7a", "jev": "#eb6834", "hybrid_cdm": "#008300", "hybrid_jev": "#6250d6"}
KIND_LEGEND = (("llm_per_kpi", "LLM por KPI (atual)"), ("llm_single_call", "LLM consolidado"),
               ("llm_json_schema", "LLM structured outputs"), ("cdm", "CDM ScaleDown"), ("jev", "Jev"),
               ("hybrid_cdm", "Híbrido CDM + LLM"), ("hybrid_jev", "Híbrido Jev + LLM"))
COMPONENT_COLORS = {"entrada (preço cheio)": "#3987e5", "entrada lida do cache": "#9ec5f4",
                    "escrita de cache": "#cde2fb", "saída": "#eb6834", "raciocínio": "#f2b28f"}
DECISION_COLORS = {"melhora": GOOD, "equivalente": "#2a78d6", "nao_inferior": "#2a78d6", "inconclusivo": MUTED,
                   "piora": CRITICAL, "sem_dados": MUTED}


def _color(summary: pd.DataFrame, b: str) -> str:
    return KIND_COLORS.get(summary.set_index("backend").at[b, "tipo"], MUTED)


def _log_usd_axis(ax, lo: float, hi: float) -> None:
    ax.set_xscale("log")
    ax.set_xlim(lo / 1.8, hi * 1.8)
    ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: usd(v)))


# =============================================================================== qualidade × custo
def _place_labels(fig, ax, items: list[dict]) -> None:
    """Posicionamento guloso de rótulos: para cada item tenta posições candidatas e fica com a primeira que não
    colide com rótulos já colocados nem com os marcadores, e que cabe dentro da área do gráfico."""
    fig.canvas.draw()
    rnd = fig.canvas.get_renderer()
    axbox = ax.get_window_extent(rnd)
    taken = [ax.transData.transform((it["x"], it["y"])) for it in items]
    marker_boxes = [matplotlib.transforms.Bbox.from_extents(px - 11, py - 11, px + 11, py + 11) for px, py in taken]
    boxes = list(marker_boxes)
    cands = [(12, 0, "left", "center"), (12, 18, "left", "bottom"), (12, -18, "left", "top"),
             (-12, 0, "right", "center"), (-12, 18, "right", "bottom"), (-12, -18, "right", "top"),
             (0, 20, "center", "bottom"), (0, -20, "center", "top"), (12, 40, "left", "bottom"),
             (12, -40, "left", "top"), (-12, 40, "right", "bottom"), (-12, -40, "right", "top")]
    for it in sorted(items, key=lambda d: -d.get("priority", 0)):
        chosen = None
        for dx, dy, ha, va in cands:
            ann = ax.annotate(it["text"], (it["x"], it["y"]), xytext=(dx, dy), textcoords="offset points",
                              ha=ha, va=va, fontsize=it.get("fontsize", 8.5), color=it.get("color", INK), zorder=5)
            bb = ann.get_window_extent(rnd).expanded(1.04, 1.12)
            inside = axbox.x0 <= bb.x0 and bb.x1 <= axbox.x1 and axbox.y0 <= bb.y0 and bb.y1 <= axbox.y1
            if inside and not any(bb.overlaps(b) for b in boxes):
                chosen = bb
                break
            ann.remove()
        if chosen is None:   # sem posição livre: usa a primeira candidata
            dx, dy, ha, va = cands[0]
            ann = ax.annotate(it["text"], (it["x"], it["y"]), xytext=(dx, dy), textcoords="offset points", ha=ha,
                              va=va, fontsize=it.get("fontsize", 8.5), color=it.get("color", INK), zorder=5)
            chosen = ann.get_window_extent(rnd)
        boxes.append(chosen)


def quality_cost_frontier(summary: pd.DataFrame, footer: str = ""):
    """Qualidade nos KPIs comuns (IC 95%) × custo projetado por 1.000 ligações, com a fronteira de eficiência."""
    s = summary.dropna(subset=["usd_1k_projetado", "q_comum"])
    fig, ax = plt.subplots(figsize=(11.5, 6.6))
    f = s[s["status_ce"] == "fronteira"].sort_values("usd_1k_projetado")
    ax.plot(f["usd_1k_projetado"], 100 * f["q_comum"], color=INK_2, lw=1.4, ls="--", zorder=1)
    labels = []
    for r in s.itertuples():
        c = KIND_COLORS.get(r.tipo, MUTED)
        dominated = str(r.status_ce).startswith("dominado")
        ax.errorbar(r.usd_1k_projetado, 100 * r.q_comum,
                    yerr=[[100 * (r.q_comum - r.q_comum_ci_low)], [100 * (r.q_comum_ci_high - r.q_comum)]],
                    fmt="o", ms=12, color=c, mfc=SURFACE if r.preco_estimado else c, mec=c, mew=2.2,
                    elinewidth=1.2, capsize=4, alpha=0.55 if dominated else 1, zorder=3)
        cov = "" if r.cobertura_pct >= 1 else f" · cobre {int(r.kpis_cobertos)} KPIs"
        labels.append({"x": r.usd_1k_projetado, "y": 100 * r.q_comum, "priority": 0 if dominated else 1,
                       "text": f"{r.backend}{' (dominado)' if dominated else ''}\n{usd(r.usd_1k_projetado)}/1k · "
                               f"{br(100 * r.q_comum, 1)}%{cov}", "color": MUTED if dominated else INK})
    _log_usd_axis(ax, s["usd_1k_projetado"].min(), s["usd_1k_projetado"].max() * 2.5)
    lo, hi = 100 * s["q_comum_ci_low"].min(), 100 * s["q_comum_ci_high"].max()
    ax.set_ylim(max(0, lo - 8), min(100, hi + 8))
    ax.yaxis.set_major_formatter(_br_axis(0, suffix="%"))
    ax.set_xlabel("Custo projetado em produção por 1.000 ligações (escala log)")
    ax.set_ylabel("Qualidade nos KPIs comuns (0% = acaso)")
    _header(ax, "Qualidade × custo por backend",
            "Canto superior esquerdo é melhor · tracejado = fronteira de eficiência · ponto vazado = preço estimado")
    ax.legend(handles=[Patch(color=KIND_COLORS[k], label=lab) for k, lab in KIND_LEGEND if k in set(s["tipo"])],
              loc="lower right", frameon=True, facecolor=SURFACE, edgecolor=GRID, fontsize=8.5)
    _footer(fig, footer)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    # ICERs da fronteira entram como rótulos de menor prioridade (no meio de cada segmento)
    for a, b in zip(f.iloc[:-1].itertuples(), f.iloc[1:].itertuples()):
        labels.append({"x": float(np.sqrt(a.usd_1k_projetado * b.usd_1k_projetado)), "y": 50 * (a.q_comum + b.q_comum),
                       "priority": -1, "text": f"ICER US$ {br(b.icer_usd_1k_por_pp, 4)}/pp", "color": INK_2,
                       "fontsize": 8})
    _place_labels(fig, ax, labels)
    return fig


def cost_per_quality(summary: pd.DataFrame, footer: str = ""):
    """US$ por 1.000 ligações por ponto de qualidade (menor é melhor)."""
    s = summary.dropna(subset=["usd_1k_por_pp"]).sort_values("usd_1k_por_pp")
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(s) + 1.8))
    y = np.arange(len(s))
    ax.barh(y, s["usd_1k_por_pp"], color=[KIND_COLORS.get(t, MUTED) for t in s["tipo"]], height=0.6, zorder=2)
    for i, r in enumerate(s.itertuples()):
        ax.text(r.usd_1k_por_pp, i, f"  US$ {br(r.usd_1k_por_pp, 5)}", va="center", fontsize=9)
    ax.set_yticks(y, s["backend"])
    ax.invert_yaxis()
    ax.set_xlim(0, s["usd_1k_por_pp"].max() * 1.35)
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(_br_axis(4, prefix="US$ "))
    ax.set_xlabel("US$ projetado por 1.000 ligações ÷ pontos de qualidade (KPIs comuns)")
    _header(ax, "Custo por ponto de qualidade", "Menor é melhor")
    _footer(fig, footer)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


# =============================================================================== custo
def cost_bars(summary: pd.DataFrame, monthly_calls: int, footer: str = ""):
    s = summary.sort_values("usd_1k_projetado")
    fig, ax = plt.subplots(figsize=(11, 0.62 * len(s) + 2))
    y = np.arange(len(s))
    h = 0.38
    for i, r in enumerate(s.itertuples()):
        c = KIND_COLORS.get(r.tipo, MUTED)
        ax.barh(i - h / 2, r.usd_1k_medido, height=h * 0.92, color=c, alpha=0.45, zorder=2)
        ax.barh(i + h / 2, r.usd_1k_projetado, height=h * 0.92, color=c, zorder=2)
        ax.text(r.usd_1k_medido, i - h / 2, f"  medido {usd(r.usd_1k_medido)}", va="center", fontsize=8.5, color=INK_2)
        ax.text(r.usd_1k_projetado, i + h / 2, f"  projetado {usd(r.usd_1k_projetado)} · {usd(r.usd_mes_projetado)}/mês",
                va="center", fontsize=8.8, color=INK, weight="bold")
    ax.set_yticks(y, s["backend"])
    ax.invert_yaxis()
    ax.set_xlim(0, s[["usd_1k_medido", "usd_1k_projetado"]].to_numpy().max() * 1.7)
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(_br_axis(2, prefix="US$ "))
    ax.set_xlabel("US$ para extrair todos os KPIs cobertos de 1.000 ligações")
    _header(ax, "Custo por 1.000 ligações", f"Barra clara = medido nesta amostra · escura = projetado "
            f"({br(monthly_calls, 0)} ligações/mês)")
    _footer(fig, footer)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def cost_breakdown_chart(breakdown: pd.DataFrame, footer: str = ""):
    """Cálculo do custo: US$ por 1.000 ligações empilhado por componente (tokens × preço) e modelo."""
    b = breakdown[breakdown["usd_1k_ligacoes"] > 0]
    order = list(dict.fromkeys(b.sort_values("usd_1k_ligacoes", ascending=False)["backend"]))
    order = sorted(order, key=lambda x: b.loc[b["backend"] == x, "usd_1k_ligacoes"].sum())
    fig, ax = plt.subplots(figsize=(11.5, 0.62 * len(order) + 2.2))
    totals = []
    for i, bk in enumerate(order):
        left = 0.0
        for r in b[b["backend"] == bk].itertuples():
            ax.barh(i, r.usd_1k_ligacoes, left=left, height=0.58, color=COMPONENT_COLORS.get(r.componente, MUTED),
                    edgecolor=SURFACE, linewidth=1, zorder=2,
                    hatch="//" if r.modelo != b[b["backend"] == bk]["modelo"].iloc[0] else None)
            left += r.usd_1k_ligacoes
        totals.append(left)
        reqs = b.loc[b["backend"] == bk].groupby("modelo")["requisicoes_por_ligacao"].first().sum()
        ax.text(left, i, f"  {usd(left)}/1k · {br(reqs, 0)} req/ligação", va="center", fontsize=9.2, weight="bold")
    ax.set_yticks(range(len(order)), order)
    ax.set_xlim(0, max(totals) * 1.45 if totals else 1)
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(_br_axis(2, prefix="US$ "))
    ax.set_xlabel("US$ por 1.000 ligações (medido) = Σ tokens por ligação × preço por 1M ÷ 1.000")
    _header(ax, "Cálculo do custo por componente", "Cada bloco = tokens médios por ligação × preço · hachurado = "
            "segundo modelo do híbrido")
    present = [c for c in COMPONENT_COLORS if c in set(b["componente"])]
    ax.legend(handles=[Patch(color=COMPONENT_COLORS[c], label=c) for c in present], loc="lower right",
              frameon=False, ncol=len(present), fontsize=8.5)
    _footer(fig, footer)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def tokens_chart(summary: pd.DataFrame, footer: str = ""):
    s = summary.sort_values("total_tokens")
    fig, ax = plt.subplots(figsize=(11, 0.55 * len(s) + 1.9))
    y = np.arange(len(s))
    parts = [("entrada_tokens", "entrada", "#3987e5"), ("saida_tokens", "saída (inclui raciocínio)", "#eb6834")]
    left = np.zeros(len(s))
    for col, lab, c in parts:
        v = s[col].to_numpy(float)
        ax.barh(y, v, left=left, color=c, height=0.58, label=lab, zorder=2, edgecolor=SURFACE)
        left += v
    for i, r in enumerate(s.itertuples()):
        ax.text(left[i], i, f"  {br(r.total_tokens, 0)} tokens · {br(r.requisicoes_por_ligacao, 1)} req", va="center",
                fontsize=9)
    ax.set_yticks(y, s["backend"])
    ax.invert_yaxis()
    ax.set_xlim(0, left.max() * 1.4)
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(_br_axis(0))
    ax.set_xlabel("Tokens por ligação (média)")
    _header(ax, "Consumo de tokens por ligação", "O Jev cobra só a entrada; a saída dele são poucos tokens")
    ax.legend(loc="lower right", frameon=False, fontsize=8.5)
    _footer(fig, footer)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


# =============================================================================== latência
def latency_chart(summary: pd.DataFrame, sla_s: float | None = None, footer: str = ""):
    s = summary.sort_values("latencia_p50_s")
    fig, ax = plt.subplots(figsize=(11, 0.55 * len(s) + 1.9))
    y = np.arange(len(s))
    ax.barh(y, s["latencia_p50_s"], color=[KIND_COLORS.get(t, MUTED) for t in s["tipo"]], height=0.58, zorder=2)
    ax.scatter(s["latencia_p95_s"], y, marker="|", s=260, color=INK, zorder=3, label="p95")
    for i, r in enumerate(s.itertuples()):
        ax.text(max(r.latencia_p95_s, r.latencia_p50_s) * 1.08, i,
                f"p50 {br(r.latencia_p50_s, 2)} s · p95 {br(r.latencia_p95_s, 2)} s", va="center", fontsize=9)
    if sla_s:
        ax.axvline(sla_s, color=CRITICAL, lw=1.2, ls="--")
        ax.text(sla_s, -0.7, f" SLA {br(sla_s, 1)} s", color=CRITICAL, fontsize=8.5, va="bottom")
    ax.set_xscale("log")
    ax.set_xlim(s["latencia_p50_s"].min() / 2, s["latencia_p95_s"].max() * 6)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{br(v, 2 if v < 1 else 0)} s"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks(y, s["backend"])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Segundos para obter todos os KPIs de uma ligação (escala log)")
    _header(ax, "Latência por ligação", "Barra = p50 · traço = p95 (menor é melhor)")
    _footer(fig, footer)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


# =============================================================================== qualidade
def quality_heatmap(quality: pd.DataFrame, backends: list[str], footer: str = ""):
    """Qualidade normalizada por KPI e backend; vazio = KPI não coberto pelo backend."""
    q = quality[quality["backend"].isin(backends)]
    piv = q.pivot_table(index="display", columns="backend", values="q_norm").reindex(columns=backends)
    order = q.drop_duplicates("display").set_index("display")["kpi"]
    piv = piv.loc[[d for d in order.index if d in piv.index]]
    cmap = LinearSegmentedColormap.from_list("seq", ["#eef4fc", "#9cc1ee", "#2a78d6", "#0d3d7a"])
    fig, ax = plt.subplots(figsize=(1.5 * len(backends) + 4, 0.5 * len(piv) + 2))
    vals = 100 * piv.to_numpy(float)
    ax.imshow(np.ma.masked_invalid(vals), cmap=cmap, vmin=0, vmax=100, aspect="auto")
    ax.grid(False)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            ax.text(j, i, "não cobre" if not np.isfinite(v) else f"{br(v, 0)}%", ha="center", va="center",
                    fontsize=8.5, color=MUTED if not np.isfinite(v) else ("#ffffff" if v > 60 else INK))
    ax.set_xticks(range(len(backends)), backends, rotation=20, ha="right")
    ax.set_yticks(range(len(piv)), piv.index)
    for sp in ax.spines.values():
        sp.set_visible(False)
    _header(ax, "Qualidade por KPI e backend", "Qualidade normalizada (0% = acaso, 100% = concordância total com a referência)")
    _footer(fig, footer)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def diff_forest(comparisons: pd.DataFrame, summary: pd.DataFrame, baseline: str, footer: str = ""):
    """Diferença na métrica primária contra a base, por KPI e backend, com IC 95% pareado."""
    c = comparisons.copy()
    kpis = list(dict.fromkeys(c["display"]))
    backs = list(dict.fromkeys(c["backend"]))
    fig, ax = plt.subplots(figsize=(11, 0.62 * len(kpis) + 2.4))
    step = 0.7 / max(len(backs), 1)
    for i, k in enumerate(kpis):
        m = c.loc[c["display"] == k, "margem"].iloc[0]
        ax.plot([-m, m], [i, i], color=GRID, lw=10, solid_capstyle="butt", zorder=1)
        for j, b in enumerate(backs):
            r = c[(c["display"] == k) & (c["backend"] == b)]
            if r.empty:
                continue
            r = r.iloc[0]
            yy = i - 0.35 + step * (j + 0.5)
            col = _color(summary, b)
            ax.plot([r["diff_ci_low"], r["diff_ci_high"]], [yy, yy], color=col, lw=1.6, zorder=2)
            ax.scatter(r["diff"], yy, color=col, s=26, zorder=3,
                       marker="D" if r["decisao"] in ("melhora", "piora") else "o")
    ax.axvline(0, color=AXIS, lw=1)
    ax.set_yticks(range(len(kpis)), kpis)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(_br_axis(2))
    ax.set_xlabel(f"Diferença na métrica primária ({'backend'} − {baseline}); IC 95% bootstrap pareado")
    _header(ax, f"Qualidade por KPI contra a base ({baseline})",
            "À direita do zero = concorda mais com a referência · faixa cinza = margem do KPI · losango = significativo")
    ax.legend(handles=[Line2D([], [], color=_color(summary, b), marker="o", lw=1.6, label=b) for b in backs],
              loc="lower right", frameon=True, facecolor=SURFACE, edgecolor=GRID, fontsize=8.5)
    _footer(fig, footer)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def calibration_chart(calibration: pd.DataFrame, threshold: float | None = None, footer: str = ""):
    """Confiança informada pelo backend × acerto contra a referência, e a curva de limiar (cobertura × acerto)."""
    if calibration.empty:
        return None
    f = calibration[(calibration["tipo_linha"] == "faixa") & (calibration["n"] > 0)]
    cur = calibration[calibration["tipo_linha"] == "limiar"]
    backs = [b for b in dict.fromkeys(f["backend"]) if "hibrido" not in b] or list(dict.fromkeys(f["backend"]))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    width = 0.8 / len(backs)
    faixas = list(dict.fromkeys(f["faixa"].astype(str)))
    for j, b in enumerate(backs):
        d = f[f["backend"] == b]
        d = d.set_index(d["faixa"].astype(str)).reindex(faixas)
        x = np.arange(len(faixas)) + (j - (len(backs) - 1) / 2) * width
        ax1.bar(x, 100 * d["acerto"].fillna(0), width=width * 0.92, color=KIND_COLORS.get(b, MUTED), zorder=2,
                label=b)
        for xi, (a, n) in enumerate(zip(d["acerto"], d["n"])):
            if np.isfinite(a):
                ax1.text(x[xi], 100 * a, f"{br(100 * a, 0)}%\nn={int(n)}", ha="center", va="bottom", fontsize=7.5)
    ax1.legend(loc="upper left", frameon=False, fontsize=8.5)
    ax1.set_xticks(range(len(faixas)), faixas)
    ax1.set_ylim(0, 118)
    ax1.yaxis.set_major_formatter(_br_axis(0, suffix="%"))
    ax1.set_xlabel("Faixa de confiança informada")
    ax1.grid(axis="x", visible=False)
    _header(ax1, "Calibração: confiança × acerto", "Acerto contra a referência, KPIs estruturados")
    for b in backs:   # cor = backend; estilo = medida
        d = cur[cur["backend"] == b]
        c = KIND_COLORS.get(b, MUTED)
        ax2.plot(d["limiar"], 100 * d["cobertura"], color=c, lw=1.4, ls=":")
        ax2.plot(d["limiar"], 100 * d["acerto_acima"], color=c, lw=1.8)
        ax2.plot(d["limiar"], 100 * d["acerto_abaixo"], color=c, lw=1.4, ls="--")
    ax2.legend(handles=[Line2D([], [], color=INK_2, ls=":", label="respostas acima do limiar"),
                        Line2D([], [], color=INK_2, label="acerto acima do limiar"),
                        Line2D([], [], color=INK_2, ls="--", label="acerto abaixo do limiar")]
               + [Patch(color=KIND_COLORS.get(b, MUTED), label=b) for b in backs],
               loc="lower left", frameon=False, fontsize=8.5)
    if threshold is not None:
        ax2.axvline(threshold, color=AXIS, lw=1, ls=":")
        ax2.text(threshold, 3, f" limiar atual {br(threshold, 2)}", fontsize=8.5, color=INK_2)
    ax2.set_ylim(0, 105)
    ax2.yaxis.set_major_formatter(_br_axis(0, suffix="%"))
    ax2.xaxis.set_major_formatter(_br_axis(2))
    ax2.set_xlabel("Limiar de confiança (abaixo dele, o híbrido reextrai no LLM)")
    _header(ax2, "Escolha do limiar do híbrido", "Abaixo do limiar o acerto cai: são as respostas a reextrair no LLM")
    _footer(fig, footer)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


# =============================================================================== tabelas
_TABLE_STYLES = [{"selector": "th", "props": [("text-align", "left"), ("background", "#f3f2ee"), ("color", INK)]},
                 {"selector": "td", "props": [("padding", "4px 10px")]}]


def style_summary(summary: pd.DataFrame):
    cols = {"backend": "Backend", "descricao": "Tipo", "kpis_cobertos": "KPIs", "requisicoes_por_ligacao": "Req/ligação",
            "usd_1k_projetado": "US$/1k projetado", "usd_mes_projetado": "US$/mês", "latencia_p50_s": "p50 (s)",
            "latencia_p95_s": "p95 (s)", "q_comum": "Qualidade (KPIs comuns)", "diff_base_pp": "Δ vs base (pp)",
            "decisao_vs_base": "Decisão", "usd_1k_por_pp": "US$/1k por pp", "status_ce": "Fronteira",
            "ligacoes_completas_pct": "Ligações completas", "preco_estimado": "Preço estimado"}
    t = summary[[c for c in cols if c in summary]].rename(columns=cols).copy()
    t["Preço estimado"] = t["Preço estimado"].map({True: "sim", False: "não"})
    pct = lambda v: f"{br(100 * v, 1)}%"  # noqa: E731
    fmt = {"Req/ligação": lambda v: br(v, 1), "US$/1k projetado": usd, "US$/mês": usd, "p50 (s)": lambda v: br(v, 2),
           "p95 (s)": lambda v: br(v, 2), "Qualidade (KPIs comuns)": pct, "Δ vs base (pp)": lambda v: br(v, 1),
           "US$/1k por pp": lambda v: br(v, 5), "Ligações completas": pct}
    return t.style.format(fmt, na_rep="—").hide(axis="index").set_table_styles(_TABLE_STYLES)
