"""Extração consolidada (1 chamada por ligação, transcrição completa): prompt, parser e estudo simulado."""
import json

import numpy as np
import pandas as pd
import pytest

from kpibench import live
from kpibench import single_call as sc
from kpibench.config import PROJECT_ROOT, load_config
from kpibench.extraction.parser import parse_all_kpis_output
from kpibench.llm.tokens import estimate_tokens
from kpibench.prompts.renderer import PromptRenderer


@pytest.fixture(scope="module")
def cfg():
    return load_config(PROJECT_ROOT / "config" / "demo.yaml")


def test_prompt_has_every_kpi_rubric_and_output_key(cfg):
    r = PromptRenderer(cfg.root / "prompts")
    kpis = sc.single_call_kpis(cfg, True)
    p = sc.render_single_call(r, kpis, "ATENDENTE: bom dia.\nCLIENTE: minha internet caiu.", "Resumo curto.")
    for k in kpis:
        assert f'"{k.name}":' in p.system                                     # chave no formato de saída
        assert sc.kpi_rubric(r, k).splitlines()[0] in p.system                # rubrica do template por KPI
        if k.labels:
            assert all(f'"{lab}"' in p.system for lab in k.labels)
    assert "Formato de saída (JSON)" not in p.system                          # blocos de formato por KPI removidos
    assert "minha internet caiu" in p.user and "<resumo>" in p.user           # resumo só por causa de qualidade_resumo
    assert estimate_tokens(p.system) >= cfg.model("gpt-6-luna").min_cacheable_tokens  # prefixo cacheável
    no_summary = sc.render_single_call(r, sc.single_call_kpis(cfg, False), "t", "Resumo curto.")
    assert "<resumo>" not in no_summary.user and '"qualidade_resumo"' not in no_summary.system


def test_prompt_hash_changes_with_kpi_template(cfg, tmp_path):
    r = PromptRenderer(cfg.root / "prompts")
    kpis = sc.single_call_kpis(cfg, True)
    assert sc.prompt_hash(r, kpis) == sc.prompt_hash(r, kpis)
    assert sc.prompt_hash(r, kpis) != sc.prompt_hash(r, kpis[:-1])


def test_parse_all_kpis_strict_per_kpi_and_tolerant_envelope(cfg):
    kpis = [cfg.kpi_registry[k] for k in ("csat", "solicitacao_resolvida", "solver_reasoning", "empatia")]
    text = "```json\n" + json.dumps({"kpis": {
        "csat": {"justificativa": "ok", "valor": "4"},
        "solicitacao_resolvida": {"justificativa": "x", "valor": "Talvez"},     # fora da taxonomia
        "solver_reasoning": "Atendente estornou a cobrança.",                  # valor direto, sem objeto
    }}, ensure_ascii=False) + "\n```"
    obj, parsed = parse_all_kpis_output(text, kpis)
    assert obj is not None
    assert parsed["csat"].valid and parsed["csat"].value == 4 and parsed["csat"].extras["justificativa"] == "ok"
    assert not parsed["solicitacao_resolvida"].valid and "rotulo_invalido" in parsed["solicitacao_resolvida"].error
    assert parsed["solver_reasoning"].valid
    assert parsed["empatia"].error.startswith("campo_ausente")
    _, bad = parse_all_kpis_output('{"kpis": {"csat": {"justificativa": "corta', kpis)
    assert all(p.error == "json_invalido" for p in bad.values())


def test_cost_table_and_savings_math(cfg):
    calls = pd.DataFrame([
        {"abordagem": sc.PER_KPI, "call_id": "1", "chamadas_llm": 10, "input_tokens": 4000, "cached_tokens": 0,
         "cache_write_tokens": 0, "output_tokens": 600, "reasoning_tokens": 0, "cost_usd": 0.0007,
         "latency_s": 20.0, "latencia_seq_s": 20.0, "latencia_paralela_s": 3.0, "prefixo_estatico_tokens": 3000,
         "conteudo_variavel_tokens": 1000, "kpis_validos": 10, "n_kpis": 10, "truncado": False, "simulated": True},
        {"abordagem": sc.SINGLE, "call_id": "1", "chamadas_llm": 1, "input_tokens": 3500, "cached_tokens": 0,
         "cache_write_tokens": 0, "output_tokens": 400, "reasoning_tokens": 0, "cost_usd": 0.00055,
         "latency_s": 6.0, "latencia_seq_s": 6.0, "latencia_paralela_s": 6.0, "prefixo_estatico_tokens": 2800,
         "conteudo_variavel_tokens": 700, "kpis_validos": 9, "n_kpis": 10, "truncado": False, "simulated": True},
    ])
    per_kpi = pd.DataFrame([{"kpi": f"k{i}", "input_tokens": 400, "cached_tokens": 0, "cache_write_tokens": 0,
                             "output_tokens": 60, "static_prefix_tokens": 300, "latency_s": 2.0} for i in range(10)])
    cost = sc.cost_table(cfg, calls, per_kpi, "gpt-6-luna", "gpt-6-luna").set_index("abordagem")
    assert cost.at[sc.PER_KPI, "usd_1k_medido"] == pytest.approx(0.7)
    assert cost.at[sc.SINGLE, "total_tokens"] == pytest.approx(3900)
    assert bool(cost.at[sc.SINGLE, "prefixo_cacheavel"]) and not bool(cost.at[sc.PER_KPI, "prefixo_cacheavel"])
    assert cost.at[sc.SINGLE, "ligacoes_completas_pct"] == 0.0
    s = sc.savings_summary(cost.reset_index(), pd.DataFrame())
    assert s["reducao_custo_medido_pct"] == pytest.approx(100 * (1 - 0.55 / 0.7))
    assert s["reducao_chamadas_pct"] == pytest.approx(90.0)
    # custo do resumo (pré-requisito do fluxo atual) entra só no fluxo atual
    c2 = sc.cost_table(cfg, calls, per_kpi, "gpt-6-luna", "gpt-6-luna", summary_cost_usd_per_call=0.001)
    assert c2.set_index("abordagem").at[sc.PER_KPI, "usd_1k_medido"] == pytest.approx(1.7)


def test_study_simulated_end_to_end(tmp_path):
    c = live.build_config(models=["gpt-6-luna"], judge_model="gpt-4o-mini", simulated=True, n_bootstrap=50)
    c.run.output_dir = str(tmp_path)
    calls = live.sample_calls(12)
    est = sc.estimate_study_cost(c, calls, model_single="gpt-6-luna", model_per_kpi="gpt-6-luna",
                                 reference="claude-opus-5-5", reference_mode="consolidado")
    assert est.iloc[-1]["papel"] == "TOTAL" and est.iloc[-1]["usd_estimado"] > 0
    st = sc.run_study(c, calls)

    n = len(st.dataset)
    assert n == 12
    by = st.calls.groupby("abordagem")["chamadas_llm"].mean()
    assert by[sc.SINGLE] == 1 and by[sc.PER_KPI] == len(c.selected_kpis)
    lines = [json.loads(x) for x in st.json_path.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == n and set(lines[0]["kpis"]) == {k.name for k in c.selected_kpis}
    # tokens/custo rateados nas predições longas somam o custo da chamada
    single_cost = st.calls.loc[st.calls["abordagem"] == sc.SINGLE, "cost_usd"].sum()
    assert st.preds.loc[st.preds["model"] == sc.SINGLE, "cost_usd"].sum() == pytest.approx(single_cost)
    assert set(st.quality["kpi"]) == {k.name for k in c.selected_kpis}
    assert st.quality["decisao"].isin(["melhora", "equivalente", "nao_inferior", "piora", "inconclusivo"]).all()
    assert st.savings["reducao_chamadas_pct"] == pytest.approx(100 * (1 - 1 / len(c.selected_kpis)))
    assert np.isfinite(st.savings["reducao_custo_medido_pct"])
    assert "Custo" in sc.executive_summary(st)
    for f in ("comparacao_qualidade.csv", "comparacao_custo_tokens.csv", "tokens_custo_por_ligacao.csv",
              "predicoes.csv", "estudo.json"):
        assert (st.out_dir / f).exists()

    # relatório final: o cálculo por componente reproduz o custo medido
    bd = sc.cost_breakdown(c, st)
    measured = st.calls.groupby("abordagem")["cost_usd"].mean() * 1000
    for ap, total in bd.groupby("abordagem")["usd_1k_ligacoes"].sum().items():
        assert total == pytest.approx(measured[ap], rel=1e-9)
    assert bd.groupby("abordagem")["pct_do_custo"].sum().round(6).eq(100).all()
    aq = sc.aggregate_quality(c, st).set_index("abordagem")
    for ap in (sc.PER_KPI, sc.SINGLE):
        assert aq.at[ap, "ci_low"] <= aq.at[ap, "q_norm"] + 1e-9 <= aq.at[ap, "ci_high"] + 2e-9
    qc = sc.quality_cost_summary(c, st)
    assert qc["veredito"] in {"domina", "troca_economia", "troca_qualidade", "dominado"}
    assert qc["usd_1k_por_pp_novo"] == pytest.approx(qc["usd_1k_novo"] / (100 * qc["q_novo"]))
    rep = sc.final_report(c, st)
    for sec in ("## 1. Diferença de custo", "## 2. Cálculo do custo", "## 3. Relação qualidade × custo",
                "**Veredito:**", "RODADA SIMULADA"):
        assert sec in rep
    pk = sc.per_kpi_cost_quality(st)
    assert len(pk) == len(st.quality) and (pk["usd_1k_novo"] > 0).all()

    from kpibench.reporting import single_call_charts as ch

    for i, f in enumerate([ch.quality_cost_dashboard(qc, pk), ch.cost_calculation(bd)]):
        f.savefig(tmp_path / f"final_{i}.png")
    ch.style_breakdown(bd).to_html()

    figs = [ch.cost_comparison(st.cost, st.savings), ch.token_breakdown(st.cost),
            ch.monthly_projection(st.cost, st.savings, 1000), ch.tokens_vs_transcript(st.calls),
            ch.latency_comparison(st.cost), ch.quality_forest(st.quality), ch.quality_by_kpi(st.quality),
            ch.cost_quality_tradeoff(st.cost, st.savings)]
    for i, f in enumerate(figs):
        f.savefig(tmp_path / f"{i}.png")
    ch.style_cost(st.cost).to_html()
    ch.style_quality(st.quality).to_html()
