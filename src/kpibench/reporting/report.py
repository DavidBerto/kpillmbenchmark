"""Relatório Markdown da rodada."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from kpibench.config import BenchmarkConfig
from kpibench.data.loader import Dataset
from kpibench.metrics.kpi_eval import BASELINE_NAME, KPIEvaluation


def _md(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    return df.to_markdown(index=False, floatfmt=floatfmt, missingval="—") if not df.empty else "_(vazio)_"


def _usd(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"US$ {x:,.2f}" if x >= 1 else f"US$ {x:.4f}"


def build_report(cfg: BenchmarkConfig, ds: Dataset, evals: dict[str, KPIEvaluation], full: pd.DataFrame,
                 rec: pd.DataFrame, plans: pd.DataFrame, costs: pd.DataFrame, meta: pd.DataFrame | None,
                 prompt_hashes: dict[str, str], overhead: dict, figures: dict[str, Path | None],
                 out_dir: Path) -> Path:
    simulated = bool(costs["simulated"].any()) if "simulated" in costs else False
    pj = cfg.projection
    L: list[str] = []
    L.append(f"# Benchmark de LLMs por KPI — {cfg.run.name}")
    L.append(f"_Gerado em {datetime.now():%Y-%m-%d %H:%M} · {len(ds.df)} chamadas · "
             f"{len(cfg.models)} modelos · {len(evals)} KPIs avaliados_\n")
    if simulated:
        L.append("> ⚠️ **RODADA SIMULADA (demo).** As saídas dos modelos foram geradas pelo provider mock a partir "
                 "dos rótulos + ruído. Tokens vêm dos prompts reais e preços de `config/models.yaml`; métricas lexicais "
                 "de resumo são calculadas de verdade. **Não use estes números para escolher modelo.**\n")
    for w in ds.warnings:
        L.append(f"> ℹ️ {w}")
    if ds.warnings:
        L.append("")

    # ------------------------------------------------------------------ sumário executivo
    L.append("## 1. Sumário executivo\n")
    L.append(f"Projeção para **{pj.monthly_calls:,} chamadas/mês**, "
             f"{'Batch API' if pj.use_batch else 'API síncrona'}, cache hit {pj.cache_hit_ratio:.0%} no prefixo "
             "estático (quando cacheável). Recomendado = modelo mais barato estatisticamente **não-inferior** ao "
             "melhor (IC 95% bootstrap pareado da diferença < margem do KPI). **Candidato econômico** = mais barato "
             "que não foi declarado inferior e cuja estimativa pontual está dentro da margem: é a hipótese de "
             "economia a confirmar com mais amostra antes de ir para produção.\n")
    r = rec.copy()
    view = pd.DataFrame({
        "KPI": r["display"], "métrica": r["metrica"],
        "recomendado": r["recomendado"], "perf": r["perf_recomendado"], "IC 95%": r["ci_recomendado"],
        "US$/1k": r["usd_1k_recomendado"], "melhor": r["melhor"], "perf melhor": r["perf_melhor"],
    })
    if r["perf_producao_atual"].notna().any():
        view["produção atual"] = r["perf_producao_atual"]
    L.append(_md(view))
    L.append("\n**Candidatos econômicos a validar** (não declarados inferiores; ponto dentro da margem):\n")
    econ = pd.DataFrame({
        "KPI": r["display"], "candidato": r["candidato_economico"], "decisão": r["decisao_candidato"],
        "perf": r["perf_candidato"], "US$/1k": r["usd_1k_candidato"],
        "economia vs recomendado %": 100 * (1 - r["usd_1k_candidato"] / r["usd_1k_recomendado"]),
        "comparações inconclusivas": r["n_inconclusivos"],
    })
    L.append(_md(econ, floatfmt=".3f"))
    L.append("\n**Planos de custo mensal** (o roteamento por KPI vs. um único modelo para tudo):\n")
    p = plans.copy()
    p["usd_mes"] = p["usd_mes"].map(_usd)
    L.append(_md(p.rename(columns={"usd_mes": "custo/mês", "kpis_nao_inferiores": "KPIs não-inferiores",
                                   "perf_media": "perf. média"})))
    n_inc = int(rec["n_inconclusivos"].sum())
    if n_inc:
        L.append(f"\n> **{n_inc} comparações inconclusivas**: o IC da diferença cruza a margem. Com "
                 f"n={len(ds.df)}, a amostra não separa modelos próximos. Ver seção 6 para o tamanho de amostra.")
    if figures.get("pareto"):
        L.append(f"\n![Pareto]({figures['pareto'].name})\n")

    # ------------------------------------------------------------------ setup
    L.append("## 2. Desenho do experimento\n")
    L.append("- **Prompt único por KPI**: o mesmo template (`prompts/kpis/<kpi>.md`) é renderizado para todos os "
             "modelos; só muda o cliente. Hash do template registrado em cada predição.")
    L.append("- **Parâmetros fixos por modelo** (`config/models.yaml`): temperatura 0 quando suportada; "
             "modelos de raciocínio com `reasoning_effort` declarado. Saída JSON estrita.")
    L.append(f"- **Saídas inválidas** (JSON quebrado, rótulo fora da taxonomia, nota fora da escala): política "
             f"`{cfg.run.invalid_policy}` — contam como erro, não são descartadas.")
    L.append("- **Métricas primárias**: ordinais → QWK (kappa quadrático); categóricos → macro-F1; "
             "texto livre → nota do LLM-juiz normalizada 0–1 (juiz: "
             f"`{cfg.judge.model if cfg.judge.enabled else 'desligado'}`), com chrF++/ROUGE-L como secundárias.")
    L.append(f"- **Estatística**: bootstrap não-paramétrico pareado, {cfg.statistics.n_bootstrap} reamostragens, "
             f"IC {cfg.statistics.ci:.0%}.\n")
    L.append("| KPI | template | hash |\n|---|---|---|")
    for k in cfg.selected_kpis:
        L.append(f"| {k.name} | `prompts/kpis/{k.prompt}` | `{prompt_hashes.get(k.name, '')}` |")
    L.append("")

    # ------------------------------------------------------------------ por KPI
    L.append("## 3. Resultados por KPI\n")
    if figures.get("heatmap"):
        L.append(f"![Heatmap]({figures['heatmap'].name})\n")
    for name, ev in evals.items():
        t = full[full["kpi"] == name].copy()
        L.append(f"### {ev.kpi.display} (`{name}` · {ev.kpi.type} · n={ev.n_items} · margem NI={ev.kpi.ni_margin})\n")
        sec = {"ordinal": ["spearman", "mae", "exact_acc", "within1_acc"],
               "categorical": ["accuracy", "kappa"],
               "text": ["chrf", "rouge_l", "tfidf_cos", "judge_coverage"]}[ev.kpi.type]
        cols = ["model", "primary", "ci_low", "ci_high"] + [c for c in sec if c in t.columns] + \
               ["valid_rate", "decision", "diff_ci_low", "diff_ci_high", "proj_usd_per_1k", "latency_p95_s", "pareto"]
        tv = t[[c for c in cols if c in t.columns]].rename(columns={"primary": ev.primary_name})
        L.append(_md(tv))
        if ev.per_class is not None:
            best = rec.loc[rec["kpi"] == name, "recomendado"].iloc[0]
            pc = ev.per_class[ev.per_class["model"] == best][["label", "f1", "support"]]
            L.append(f"\nF1 por classe — `{best}`:\n")
            L.append(_md(pc))
        L.append("")

    # ------------------------------------------------------------------ qualidade do resumo
    L.append("## 4. Qualidade do resumo — métricas modernas vs. eficientes\n")
    if meta is None or meta.empty:
        L.append("_Suíte de métricas não executada._\n")
    else:
        L.append("Cada métrica é correlacionada com a nota humana (`gt_qualidade_resumo`). "
                 "`auc_detecta_ruim` mede a capacidade de separar resumos com nota ≤ 2 (uso de triagem). "
                 "Custos: métricas de CPU são ~US$0; modelos locais têm custo de infra (ver ms/item); "
                 "LLM-juiz tem custo de API medido.\n")
        cols = [c for c in ["metric", "grupo", "familia", "referencia", "spearman", "spearman_ci_low",
                            "spearman_ci_high", "kendall_tau_b", "auc_detecta_ruim", "cobertura", "ms_por_item",
                            "usd_por_1k", "status"] if c in meta.columns]
        L.append(_md(meta[cols]))
        if figures.get("meta"):
            L.append(f"\n![Meta-avaliação]({figures['meta'].name})\n")
        ok = meta[meta["status"] == "ok"]
        if "spearman" in ok.columns and ok["spearman"].notna().any():
            best_eff = ok[ok["grupo"] == "eficiente"].sort_values("spearman", ascending=False).head(1)
            best_mod = ok[ok["grupo"] == "moderna"].sort_values("spearman", ascending=False).head(1)
            if len(best_eff) and len(best_mod):
                be, bm = best_eff.iloc[0], best_mod.iloc[0]
                L.append(f"**Leitura:** melhor eficiente = `{be['metric']}` (ρ={be['spearman']:.3f}); "
                         f"melhor moderna = `{bm['metric']}` (ρ={bm['spearman']:.3f}, "
                         f"{_usd(bm['usd_por_1k'])}/1k medido, API síncrona). Ganho de concordância de "
                         f"{bm['spearman'] - be['spearman']:+.3f} em ρ. Arquitetura sugerida: métrica eficiente "
                         "como filtro de 100% do volume + juiz moderno em amostra/estratos sinalizados.\n")
        skipped = meta[meta["status"] != "ok"]
        if len(skipped):
            L.append("Métricas puladas nesta rodada (instale `.[neural]` e garanta acesso ao Hugging Face Hub):\n")
            for s in skipped.itertuples():
                L.append(f"- `{s.metric}` — {s.status}")
            L.append("")

    # ------------------------------------------------------------------ custos
    L.append("## 5. Custo e latência\n")
    agg = costs.groupby("model").agg(
        tier=("tier", "first"),
        tokens_entrada=("mean_input_tokens", "mean"), tokens_saida=("mean_output_tokens", "mean"),
        usd_1k_medido=("measured_usd_per_1k", "mean"), usd_1k_std=("proj_usd_per_1k_standard", "mean"),
        usd_1k_batch=("proj_usd_per_1k_batch", "mean"), usd_mes_todos_kpis=("proj_usd_month", "sum"),
        lat_p50=("latency_p50_s", "median"), lat_p95=("latency_p95_s", "max"),
        erro=("error_rate", "mean"), prefixo_cacheavel=("prefix_cacheable", "any"),
    ).reset_index().sort_values("usd_mes_todos_kpis")
    L.append("Médias por chamada de KPI (um KPI por chamada). `usd_mes_todos_kpis` = 10 chamadas por ligação.\n")
    L.append(_md(agg, floatfmt=".4f"))
    if not costs["prefix_cacheable"].any():
        L.append("\n> Nenhum system prompt atinge o mínimo cacheável (≈1.024 tokens): prompt caching não reduz custo "
                 "neste desenho. Não vale inflar o prompt só para cachear — o ganho real está na Batch API.")
    L.append(f"\nCusto de avaliação desta rodada (juízes, não entra em produção): "
             f"juiz de justificativas {_usd(overhead['judge_usd'])}, fidelidade por afirmações {_usd(overhead['claims_usd'])}.\n")

    # ------------------------------------------------------------------ validade
    L.append("## 6. Limitações e ameaças à validade\n")
    worst = full[full["model"] != BASELINE_NAME].assign(w=lambda d: d["ci_high"] - d["ci_low"])["w"].median()
    n = len(ds.df)
    target = max(cfg.kpi_registry[k].ni_margin for k in evals) if evals else 0.05
    n_needed = int(np.ceil(n * (worst / max(target, 1e-6)) ** 2)) if np.isfinite(worst) else None
    L.append(f"- **Tamanho de amostra**: largura mediana do IC 95% = {worst:.3f} com n={n}. "
             + (f"Para ICs com largura ≈ margem ({target}), n ≈ {n_needed:,} (escala ~1/√n). " if n_needed else "")
             + "Estratifique por motivo de ligação e desfecho.")
    L.append("- **Ruído do rótulo**: os rótulos validados também têm erro; meça a concordância entre auditores "
             "(kappa inter-anotador) — é o teto prático de qualquer modelo.")
    L.append("- **Viés do LLM-juiz**: juiz da mesma família que o candidato tende a favorecê-lo (self-preference). "
             "Calibre o juiz contra uma amostra com nota humana antes de confiar na métrica de texto.")
    L.append("- **KPIs a partir do resumo**: erros do resumo propagam para todos os KPIs. Um resumo ruim limita "
             "o teto de CSAT/FCR/ODC independentemente do modelo extrator.")
    L.append("- **Preços**: conferidos nas datas em `config/models.yaml` (algumas fontes são agregadores). "
             "Valide no contrato Azure/Anthropic/Google antes de orçar.")
    L.append("- **Deriva**: reavalie ao trocar versão de modelo, prompt (hash muda) ou perfil de chamadas.\n")

    path = out_dir / "report.md"
    path.write_text("\n".join(L), encoding="utf-8")
    return path
