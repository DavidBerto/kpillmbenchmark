"""Gráficos do estudo de extração consolidada (notebook 04). Mesma linguagem visual de aa_charts:
uma pergunta por gráfico, título que afirma a métrica, rótulo direto em toda barra e ponto.

Cores: atual (1 chamada por KPI) = cinza neutro; novo (1 chamada por ligação) = azul (slot 1);
produção atual = cinza claro. Melhora/piora usam verde/vermelho só nos marcadores de decisão.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from kpibench.reporting.aa_charts import (AXIS, CRITICAL, GOOD, GRID, INK, INK_2, MUTED, NEUTRAL, SURFACE,
                                          _br_axis, _footer, _header, br, usd)
from kpibench.single_call import APPROACH_LABELS, PER_KPI, SINGLE

C_OLD = "#6f6e69"
C_NEW = "#2a78d6"
C_PROD = NEUTRAL
DECISION_COLORS = {"melhora": GOOD, "equivalente": C_NEW, "nao_inferior": C_NEW, "inconclusivo": MUTED,
                   "piora": CRITICAL, "sem_dados": MUTED}
DECISION_LABELS = {"melhora": "melhora", "equivalente": "equivalente", "nao_inferior": "não-inferior",
                   "inconclusivo": "inconclusivo", "piora": "piora", "sem_dados": "sem dados"}
TOKEN_PARTS = [("prefixo_estatico_tokens", "instruções (system prompt)", "#9ec5f4"),
               ("conteudo_variavel_tokens", "conteúdo (resumo ou transcrição)", "#3987e5"),
               ("saida_visivel_tokens", "saída (JSON)", "#eb6834"),
               ("raciocinio_tokens", "raciocínio", "#f2b28f")]


def _order(cost: pd.DataFrame) -> pd.DataFrame:
    return cost.set_index("abordagem").reindex([a for a in (PER_KPI, SINGLE) if a in set(cost["abordagem"])])


# =============================================================================== custo
def cost_comparison(cost: pd.DataFrame, savings: dict, footer: str = ""):
    """US$ por 1.000 ligações: medido nesta amostra × projetado em produção, atual × novo."""
    c = _order(cost)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    y = np.arange(2)
    h = 0.36
    for i, (col, lab, alpha) in enumerate((("usd_1k_medido", "medido nesta amostra", 1.0),
                                           ("usd_1k_projetado", "projetado em produção", 0.55))):
        vals = c[col].to_numpy(float)
        for j, ap in enumerate(c.index):
            color = C_NEW if ap == SINGLE else C_OLD
            yy = y[i] + (j - 0.5) * h
            ax.barh(yy, vals[j], height=h * 0.9, color=color, alpha=alpha, zorder=2)
            ax.text(vals[j], yy, f"  {usd(vals[j])}", va="center", fontsize=9.5, color=INK)
    ax.set_yticks(y, ["Medido\n(esta amostra)", "Projetado\n(produção)"])
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(_br_axis(2, prefix="US$ "))
    ax.set_xlim(0, float(np.nanmax(c[["usd_1k_medido", "usd_1k_projetado"]].to_numpy())) * 1.35)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("US$ para extrair todos os KPIs de 1.000 ligações")
    red_m, red_p = savings.get("reducao_custo_medido_pct", np.nan), savings.get("reducao_custo_projetado_pct", np.nan)
    _header(ax, "Custo por 1.000 ligações", f"Redução do fluxo novo: {br(red_m, 1)}% medida · {br(red_p, 1)}% projetada "
            "(menor é melhor)")
    ax.legend(handles=[Patch(color=C_OLD, label=APPROACH_LABELS[PER_KPI]), Patch(color=C_NEW, label=APPROACH_LABELS[SINGLE])],
              loc="lower right", frameon=False)
    _footer(fig, footer)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def token_breakdown(cost: pd.DataFrame, footer: str = ""):
    """Tokens por ligação, por componente (instruções, conteúdo, saída, raciocínio)."""
    c = _order(cost).copy()
    c["saida_visivel_tokens"] = (c["saida_tokens"] - c["raciocinio_tokens"].fillna(0)).clip(lower=0)
    fig, ax = plt.subplots(figsize=(10, 3.8))
    y = np.arange(len(c))
    left = np.zeros(len(c))
    for col, lab, color in TOKEN_PARTS:
        v = c[col].fillna(0).to_numpy(float)
        ax.barh(y, v, left=left, color=color, height=0.55, label=lab, zorder=2, edgecolor=SURFACE, linewidth=1)
        for i in range(len(c)):
            if v[i] > 0.06 * c["total_tokens"].max():
                ax.text(left[i] + v[i] / 2, y[i], br(v[i], 0), ha="center", va="center", fontsize=8.5, color=INK)
        left += v
    for i, ap in enumerate(c.index):
        calls = c.at[ap, "chamadas_por_ligacao"]
        ax.text(left[i], y[i], f"  {br(left[i], 0)} tokens · {br(calls, 0)} chamada(s)", va="center", fontsize=9.5,
                color=INK, weight="bold")
    ax.set_yticks(y, [APPROACH_LABELS[a] for a in c.index])
    ax.invert_yaxis()
    ax.set_xlim(0, left.max() * 1.45)
    ax.xaxis.set_major_formatter(_br_axis(0))
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Tokens por ligação (média; entrada inclui cache)")
    _header(ax, "Consumo de tokens por ligação",
            "O fluxo atual repete as instruções e o resumo a cada KPI; o novo lê a transcrição uma vez")
    ax.legend(loc="lower right", frameon=False, ncol=2, fontsize=8.5)
    _footer(fig, footer)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def monthly_projection(cost: pd.DataFrame, savings: dict, monthly_calls: int, footer: str = ""):
    """Custo mensal projetado e economia, no volume de `projection.monthly_calls`."""
    c = _order(cost)
    fig, ax = plt.subplots(figsize=(8, 4))
    vals = c["usd_mes_projetado"].to_numpy(float)
    colors = [C_NEW if a == SINGLE else C_OLD for a in c.index]
    bars = ax.bar([APPROACH_LABELS[a].replace(": ", ":\n") for a in c.index], vals, color=colors, width=0.55, zorder=2)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, usd(v), ha="center", va="bottom", fontsize=10.5, weight="bold")
    ax.set_ylim(0, np.nanmax(vals) * 1.25)
    ax.yaxis.set_major_formatter(_br_axis(0, prefix="US$ "))
    ax.grid(axis="x", visible=False)
    _header(ax, f"Custo mensal projetado ({br(monthly_calls, 0)} ligações/mês)",
            f"Economia projetada: {usd(savings.get('economia_mes_projetada_usd', np.nan))} por mês")
    _footer(fig, footer)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def tokens_vs_transcript(calls: pd.DataFrame, footer: str = ""):
    """Custo por ligação × tamanho da transcrição: o fluxo novo escala com a transcrição; o atual, com o resumo."""
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for ap, color, mk in ((PER_KPI, C_OLD, "s"), (SINGLE, C_NEW, "o")):
        g = calls[(calls["abordagem"] == ap) & calls["latency_s"].notna()]
        if g.empty:
            continue
        x, yv = g["transcricao_chars"].to_numpy(float) / 1000, g["cost_usd"].to_numpy(float) * 1000
        ax.scatter(x, yv, s=26, color=color, alpha=0.75, marker=mk, label=APPROACH_LABELS[ap], zorder=3,
                   edgecolor=SURFACE, linewidth=0.6)
        if len(g) >= 3 and np.ptp(x) > 0:
            k, b0 = np.polyfit(x, yv, 1)
            xs = np.linspace(x.min(), x.max(), 20)
            ax.plot(xs, k * xs + b0, color=color, lw=1.4, alpha=0.8)
    ax.yaxis.set_major_formatter(_br_axis(2, prefix="US$ "))
    ax.xaxis.set_major_formatter(_br_axis(0, suffix=" mil"))
    ax.set_xlabel("Tamanho da transcrição (caracteres)")
    ax.set_ylabel("US$ por 1.000 ligações deste tamanho")
    _header(ax, "Custo por ligação × tamanho da transcrição",
            "Ligações longas encarecem o fluxo novo; acima do cruzamento das retas, o atual fica mais barato")
    ax.legend(loc="upper left", frameon=False)
    _footer(fig, footer)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def latency_comparison(cost: pd.DataFrame, footer: str = ""):
    c = _order(cost)
    rows = [("Atual (chamadas em sequência)", c.at[PER_KPI, "latencia_p50_s"], c.at[PER_KPI, "latencia_p95_s"], C_OLD),
            ("Atual (chamadas em paralelo)", c.at[PER_KPI, "latencia_paralela_p50_s"], np.nan, "#a9a8a2"),
            ("Novo (1 chamada)", c.at[SINGLE, "latencia_p50_s"], c.at[SINGLE, "latencia_p95_s"], C_NEW)] \
        if PER_KPI in c.index and SINGLE in c.index else []
    fig, ax = plt.subplots(figsize=(9, 3.4))
    for i, (lab, p50, p95, color) in enumerate(rows):
        ax.barh(i, p50, color=color, height=0.55, zorder=2)
        txt = f"  p50 {br(p50, 1)} s" + (f" · p95 {br(p95, 1)} s" if np.isfinite(p95) else "")
        ax.text(p50, i, txt, va="center", fontsize=9.5)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, max([r[1] for r in rows] + [1]) * 1.6)
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(_br_axis(1, suffix=" s"))
    _header(ax, "Latência para extrair todos os KPIs de uma ligação", "p50 por ligação (menor é melhor)")
    _footer(fig, footer)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


# =============================================================================== qualidade
def quality_forest(quality: pd.DataFrame, footer: str = ""):
    """Diferença novo − atual na métrica primária por KPI, com IC 95% pareado e a margem do KPI."""
    q = quality.sort_values("diff_novo_menos_atual").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 0.48 * len(q) + 2.2))
    y = np.arange(len(q))
    for i, r in q.iterrows():
        color = DECISION_COLORS.get(r["decisao"], MUTED)
        ax.plot([-r["margem"], r["margem"]], [i, i], color=GRID, lw=7, solid_capstyle="butt", zorder=1)
        ax.plot([r["diff_ci_low"], r["diff_ci_high"]], [i, i], color=color, lw=2, zorder=2)
        ax.scatter(r["diff_novo_menos_atual"], i, color=color, s=50, zorder=3, edgecolor=SURFACE, linewidth=1)
        ax.text(1.01, i, f"{DECISION_LABELS.get(r['decisao'], r['decisao'])}  ({r['metrica']})",
                transform=ax.get_yaxis_transform(), va="center", fontsize=8.5, color=color if r["decisao"] in
                ("melhora", "piora") else INK_2)
    ax.axvline(0, color=AXIS, lw=1)
    ax.set_yticks(y, q["display"])
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(_br_axis(2, suffix=""))
    ax.set_xlabel("Diferença na métrica primária (novo − atual); IC 95% bootstrap pareado")
    _header(ax, "Qualidade: fluxo novo × fluxo atual, por KPI",
            "À direita do zero = o novo concorda mais com a referência · faixa cinza = margem de equivalência")
    ax.legend(handles=[Line2D([], [], color=DECISION_COLORS[d], marker="o", lw=2, label=DECISION_LABELS[d])
                       for d in ("melhora", "equivalente", "inconclusivo", "piora")],
              loc="lower right", frameon=True, fontsize=8.5, facecolor=SURFACE, edgecolor=GRID)
    _footer(fig, footer)
    fig.tight_layout(rect=(0, 0.03, 0.86, 1))
    return fig


def quality_by_kpi(quality: pd.DataFrame, footer: str = ""):
    """Métrica primária por KPI: atual × novo × produção (quando comparável), com IC 95%."""
    q = quality.reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(q) + 2.2))
    y = np.arange(len(q))
    for i, r in q.iterrows():
        ax.plot([r["atual"], r["novo"]], [i, i], color=GRID, lw=3, zorder=1)
        ax.errorbar(r["atual"], i - 0.12, xerr=[[r["atual"] - r["atual_ci_low"]], [r["atual_ci_high"] - r["atual"]]],
                    fmt="s", color=C_OLD, ms=7, elinewidth=1, zorder=3)
        ax.errorbar(r["novo"], i + 0.12, xerr=[[r["novo"] - r["novo_ci_low"]], [r["novo_ci_high"] - r["novo"]]],
                    fmt="o", color=C_NEW, ms=7, elinewidth=1, zorder=3)
        if np.isfinite(r["producao"]):
            ax.scatter(r["producao"], i, marker="D", s=40, color=C_PROD, edgecolor=INK_2, zorder=2)
    ax.set_yticks(y, [f"{d} ({m})" for d, m in zip(q["display"], q["metrica"])])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(_br_axis(2))
    ax.set_xlabel("Concordância com a referência (métrica primária do KPI)")
    _header(ax, "Concordância com a referência, por KPI", "Maior é melhor · barras = IC 95% bootstrap")
    ax.legend(handles=[Line2D([], [], marker="s", ls="", color=C_OLD, label=APPROACH_LABELS[PER_KPI]),
                       Line2D([], [], marker="o", ls="", color=C_NEW, label=APPROACH_LABELS[SINGLE]),
                       Line2D([], [], marker="D", ls="", color=C_PROD, markeredgecolor=INK_2, label="Produção atual")],
              loc="lower left", frameon=True, fontsize=8.5, facecolor=SURFACE, edgecolor=GRID)
    _footer(fig, footer)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def cost_quality_tradeoff(cost: pd.DataFrame, savings: dict, footer: str = ""):
    """Qualidade normalizada média × custo por 1.000 ligações: o canto superior esquerdo vence."""
    c = _order(cost)
    q = {PER_KPI: savings.get("qualidade_normalizada_atual", np.nan), SINGLE: savings.get("qualidade_normalizada_novo", np.nan)}
    fig, ax = plt.subplots(figsize=(8.5, 5))
    pts = [(c.at[a, "usd_1k_medido"], 100 * q[a], a) for a in c.index]
    ax.annotate("", xy=pts[-1][:2], xytext=pts[0][:2],
                arrowprops=dict(arrowstyle="->", color=INK_2, lw=1.3)) if len(pts) == 2 else None
    for x, yv, a in pts:
        ax.scatter(x, yv, s=160, color=C_NEW if a == SINGLE else C_OLD, zorder=3, edgecolor=SURFACE, linewidth=1.5)
        ax.annotate(f"{APPROACH_LABELS[a]}\n{usd(x)}/1k · {br(yv, 1)}%", (x, yv), xytext=(10, -4),
                    textcoords="offset points", fontsize=9, color=INK)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.set_xlim(0, max(xs) * 1.6)
    ax.set_ylim(max(0, min(ys) - 15), min(100, max(ys) + 15))
    ax.xaxis.set_major_formatter(_br_axis(2, prefix="US$ "))
    ax.yaxis.set_major_formatter(_br_axis(0, suffix="%"))
    ax.set_xlabel("US$ por 1.000 ligações (medido)")
    ax.set_ylabel("Qualidade normalizada média (0% = acaso)")
    _header(ax, "Custo × qualidade", "Seta = mudança do fluxo atual para o novo · canto superior esquerdo é melhor")
    _footer(fig, footer)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


# =============================================================================== relatório final
COMPONENT_COLORS = {"entrada (preço cheio)": "#3987e5", "entrada lida do cache": "#9ec5f4",
                    "escrita de cache": "#cde2fb", "saída (JSON)": "#eb6834", "raciocínio": "#f2b28f"}


def quality_cost_dashboard(summary: dict, per_kpi: pd.DataFrame, footer: str = ""):
    """Relação qualidade × custo em três painéis:
    (1) agregado: qualidade normalizada média (IC 95%) × US$ por 1.000 ligações, com a seta atual → novo;
    (2) por KPI: a mesma relação, KPI a KPI (custo do novo = rateio da chamada única);
    (3) custo por ponto de qualidade (US$/1k ÷ pp): menor é melhor."""
    fig = plt.figure(figsize=(15, 5.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.25, 0.8], wspace=0.32)
    ax1, ax2, ax3 = (fig.add_subplot(gs[0, i]) for i in range(3))

    # (1) agregado
    pts = [(PER_KPI, summary["usd_1k_atual"], summary["q_atual"], summary["q_atual_ci"], C_OLD, "s"),
           (SINGLE, summary["usd_1k_novo"], summary["q_novo"], summary["q_novo_ci"], C_NEW, "o")]
    ax1.annotate("", xy=(pts[1][1], 100 * pts[1][2]), xytext=(pts[0][1], 100 * pts[0][2]),
                 arrowprops=dict(arrowstyle="-|>", color=INK_2, lw=1.4, shrinkA=9, shrinkB=9))
    for ap, x, q, ci, color, mk in pts:
        ax1.errorbar(x, 100 * q, yerr=[[100 * (q - ci[0])], [100 * (ci[1] - q)]], fmt=mk, ms=11, color=color,
                     elinewidth=1.4, capsize=4, zorder=3, mec=SURFACE, mew=1.5)
        ax1.annotate(f"{APPROACH_LABELS[ap].split(':')[0]}\n{usd(x)}/1k · {br(100 * q, 1)}%", (x, 100 * q),
                     xytext=(12, 10), textcoords="offset points", fontsize=9, color=INK, va="bottom")
    xs = [p[1] for p in pts]
    lo = min(100 * p[3][0] for p in pts)
    hi = max(100 * p[3][1] for p in pts)
    ax1.set_xlim(0, max(xs) * 1.7)
    ax1.set_ylim(max(0, lo - 10), min(100, hi + 10))
    ax1.xaxis.set_major_locator(MaxNLocator(4))
    ax1.xaxis.set_major_formatter(_br_axis(2, prefix="US$ "))
    ax1.yaxis.set_major_formatter(_br_axis(0, suffix="%"))
    ax1.set_xlabel("US$ por 1.000 ligações (medido)")
    ax1.set_ylabel("Qualidade normalizada média (0% = acaso)")
    verdict = {"domina": ("novo domina: mais barato e melhor", GOOD),
               "troca_economia": ("novo mais barato, qualidade menor", INK_2),
               "troca_qualidade": ("novo mais caro, qualidade maior", INK_2),
               "dominado": ("novo dominado: mais caro e não melhor", CRITICAL)}.get(summary["veredito"], ("", INK_2))
    _header(ax1, "Todos os KPIs", f"Δ qualidade {'+' if summary['diff_pp'] >= 0 else '−'}{br(abs(summary['diff_pp']), 1)} pp · Δ custo "
            f"{'+' if summary['diff_usd_1k'] >= 0 else '−'}{usd(abs(summary['diff_usd_1k']))}/1k")
    ax1.text(0.02, 0.03, verdict[0], transform=ax1.transAxes, fontsize=9.5, color=verdict[1], weight="bold")

    # (2) por KPI
    d = per_kpi.dropna(subset=["usd_1k_atual", "usd_1k_novo", "q_atual", "q_novo"])
    for r in d.itertuples():
        color = DECISION_COLORS.get(r.decisao, MUTED)
        ax2.annotate("", xy=(r.usd_1k_novo, 100 * r.q_novo), xytext=(r.usd_1k_atual, 100 * r.q_atual),
                     arrowprops=dict(arrowstyle="-|>", color=color, lw=1.1, alpha=0.8, shrinkA=3, shrinkB=3))
        ax2.scatter(r.usd_1k_atual, 100 * r.q_atual, marker="s", s=22, color=C_OLD, zorder=3)
        ax2.scatter(r.usd_1k_novo, 100 * r.q_novo, marker="o", s=30, color=C_NEW, zorder=3)
        ax2.annotate(r.display, (r.usd_1k_atual, 100 * r.q_atual), xytext=(5, 3), textcoords="offset points",
                     fontsize=7.5, color=INK_2)
    if len(d):
        ax2.set_xlim(0, max(d["usd_1k_atual"].max(), d["usd_1k_novo"].max()) * 1.35)
    ax2.xaxis.set_major_locator(MaxNLocator(5))
    ax2.xaxis.set_major_formatter(_br_axis(3, prefix="US$ "))
    ax2.yaxis.set_major_formatter(_br_axis(0, suffix="%"))
    ax2.set_xlabel("US$ por 1.000 ligações para este KPI (novo = rateio da chamada única)")
    ax2.set_ylabel("Qualidade normalizada do KPI")
    _header(ax2, "Por KPI", "■ atual → ● novo · cor da seta = decisão estatística do KPI")
    ax2.legend(handles=[Line2D([], [], color=DECISION_COLORS[k], lw=2, label=DECISION_LABELS[k])
                        for k in ("melhora", "equivalente", "inconclusivo", "piora")],
               loc="lower right", fontsize=8, frameon=True, facecolor=SURFACE, edgecolor=GRID)

    # (3) custo por ponto de qualidade
    vals = [summary["usd_1k_por_pp_atual"], summary["usd_1k_por_pp_novo"]]
    labels = ["Atual\n(1 por KPI)", "Novo\n(consolidado)"]
    bars = ax3.bar(labels, vals, color=[C_OLD, C_NEW], width=0.55, zorder=2)
    for bar, v in zip(bars, vals):
        ax3.text(bar.get_x() + bar.get_width() / 2, v, f"US$ {br(v, 4)}", ha="center", va="bottom", fontsize=9.5,
                 weight="bold")
    ax3.set_ylim(0, np.nanmax(vals) * 1.3 if np.isfinite(np.nanmax(vals)) else 1)
    ax3.yaxis.set_major_formatter(_br_axis(3, prefix="US$ "))
    ax3.grid(axis="x", visible=False)
    ax3.set_ylabel("US$/1k ligações por ponto de qualidade")
    red = 100 * (1 - vals[1] / vals[0]) if vals[0] else np.nan
    _header(ax3, "Custo por ponto de qualidade", f"Menor é melhor · {br(red, 1)}% {'menor' if red >= 0 else 'maior'} no novo")

    fig.suptitle("Relação qualidade × custo: uma LLM por KPI × todos os KPIs numa chamada", x=0.01, ha="left",
                 fontsize=13, weight="bold", color=INK, y=1.04)
    _footer(fig, footer)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.85, bottom=0.16)
    return fig


def cost_calculation(breakdown: pd.DataFrame, footer: str = ""):
    """Cálculo do custo: US$ por 1.000 ligações empilhado por componente (tokens × preço), por abordagem."""
    b = breakdown[breakdown["usd_1k_ligacoes"] > 0]
    order = [a for a in (PER_KPI, SINGLE) if a in set(b["abordagem"])]
    fig, ax = plt.subplots(figsize=(11, 4.4))
    totals = []
    for i, ap in enumerate(order):
        left = 0.0
        for r in b[b["abordagem"] == ap].itertuples():
            ax.barh(i, r.usd_1k_ligacoes, left=left, height=0.55, color=COMPONENT_COLORS.get(r.componente, MUTED),
                    edgecolor=SURFACE, linewidth=1, zorder=2)
            if r.pct_do_custo >= 12:
                ax.text(left + r.usd_1k_ligacoes / 2, i,
                        f"{br(r.tokens_por_ligacao, 0)} tok\n× US$ {br(r.preco_usd_por_1M, 2)}/1M",
                        ha="center", va="center", fontsize=7.8, color=INK)
            left += r.usd_1k_ligacoes
        totals.append(left)
        ax.text(left, i, f"  {usd(left)} / 1.000 ligações", va="center", fontsize=10, weight="bold", color=INK)
    ax.set_yticks(range(len(order)), [APPROACH_LABELS[a] for a in order])
    ax.invert_yaxis()
    ax.set_xlim(0, max(totals) * 1.4 if totals else 1)
    ax.xaxis.set_major_formatter(_br_axis(2, prefix="US$ "))
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("US$ por 1.000 ligações (medido) = Σ tokens por ligação × preço por 1M ÷ 1.000")
    _header(ax, "Cálculo do custo por componente", "Cada bloco = tokens médios por ligação × preço do componente")
    present = [c for c in COMPONENT_COLORS if c in set(b["componente"])]
    ax.legend(handles=[Patch(color=COMPONENT_COLORS[c], label=c) for c in present], loc="lower right",
              frameon=False, ncol=len(present), fontsize=8.5)
    _footer(fig, footer)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def style_breakdown(breakdown: pd.DataFrame):
    t = pd.DataFrame({"Abordagem": breakdown["descricao"], "Componente": breakdown["componente"],
                      "Tokens por ligação": breakdown["tokens_por_ligacao"], "US$ por 1M tokens": breakdown["preco_usd_por_1M"],
                      "US$ por ligação": breakdown["usd_por_ligacao"], "US$ por 1.000 ligações": breakdown["usd_1k_ligacoes"],
                      "% do custo": breakdown["pct_do_custo"]})
    return (t.style.format({"Tokens por ligação": lambda v: br(v, 0), "US$ por 1M tokens": lambda v: f"US$ {br(v, 3)}",
                            "US$ por ligação": lambda v: f"US$ {br(v, 6)}", "US$ por 1.000 ligações": usd,
                            "% do custo": lambda v: f"{br(v, 1)}%"})
            .hide(axis="index").set_table_styles(_TABLE_STYLES))


# =============================================================================== tabelas
_TABLE_STYLES = [{"selector": "th", "props": [("text-align", "left"), ("background", "#f3f2ee"), ("color", INK)]},
                 {"selector": "td", "props": [("padding", "4px 10px")]}]


def style_cost(cost: pd.DataFrame):
    cols = {"descricao": "Abordagem", "modelo_llm": "Modelo", "ligacoes": "Ligações",
            "chamadas_por_ligacao": "Chamadas/ligação", "entrada_tokens": "Entrada (tokens)",
            "prefixo_estatico_tokens": "Instruções (tokens)", "conteudo_variavel_tokens": "Conteúdo (tokens)",
            "saida_tokens": "Saída (tokens)", "raciocinio_tokens": "Raciocínio (tokens)",
            "total_tokens": "Total (tokens)", "usd_1k_medido": "US$/1k medido", "usd_1k_projetado": "US$/1k projetado",
            "usd_mes_projetado": "US$/mês projetado", "latencia_p50_s": "Latência p50",
            "ligacoes_completas_pct": "Ligações com todos os KPIs", "truncadas_pct": "Truncadas"}
    t = cost[[c for c in cols if c in cost]].rename(columns=cols)
    tok = lambda v: br(v, 0)  # noqa: E731
    fmt = {"Chamadas/ligação": tok, "Entrada (tokens)": tok, "Instruções (tokens)": tok, "Conteúdo (tokens)": tok,
           "Saída (tokens)": tok, "Raciocínio (tokens)": tok, "Total (tokens)": tok, "US$/1k medido": usd,
           "US$/1k projetado": usd, "US$/mês projetado": usd, "Latência p50": lambda v: f"{br(v, 1)} s",
           "Ligações com todos os KPIs": lambda v: f"{br(100 * v, 1)}%", "Truncadas": lambda v: f"{br(100 * v, 1)}%"}
    return t.style.format(fmt).hide(axis="index").set_table_styles(_TABLE_STYLES)


def style_quality(quality: pd.DataFrame):
    q = quality.copy()
    q["Atual"] = [f"{br(a, 3)} [{br(lo, 2)}; {br(hi, 2)}]" for a, lo, hi in zip(q["atual"], q["atual_ci_low"], q["atual_ci_high"])]
    q["Novo"] = [f"{br(a, 3)} [{br(lo, 2)}; {br(hi, 2)}]" for a, lo, hi in zip(q["novo"], q["novo_ci_low"], q["novo_ci_high"])]
    q["Novo − atual"] = [f"{br(d, 3)} [{br(lo, 2)}; {br(hi, 2)}]" for d, lo, hi in
                         zip(q["diff_novo_menos_atual"], q["diff_ci_low"], q["diff_ci_high"])]
    t = pd.DataFrame({"KPI": q["display"], "Métrica": q["metrica"], "n": q["n"], "Atual": q["Atual"], "Novo": q["Novo"],
                      "Produção": q["producao"], "Novo − atual": q["Novo − atual"],
                      "P(novo melhor)": q["p_novo_melhor"], "Decisão": q["decisao"].map(DECISION_LABELS),
                      "Válidas (atual)": q["atual_validas"], "Válidas (novo)": q["novo_validas"]})

    def color(v):
        return {"melhora": f"color: {GOOD}; font-weight: bold", "piora": f"color: {CRITICAL}; font-weight: bold"}.get(
            {v2: k for k, v2 in DECISION_LABELS.items()}.get(v, ""), "")
    pct = lambda v: f"{br(100 * v, 0)}%"  # noqa: E731
    return (t.style.format({"Produção": lambda v: br(v, 3), "P(novo melhor)": pct, "Válidas (atual)": pct,
                            "Válidas (novo)": pct}).hide(axis="index").set_table_styles(_TABLE_STYLES)
            .map(color, subset=["Decisão"]))
