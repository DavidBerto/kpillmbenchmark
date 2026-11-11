"""Teste de ligações: preços, extração de texto do Azure, baseline de produção e fluxo simulado."""
import numpy as np
import pandas as pd
import pytest

from kpibench import live
from kpibench.config import PROJECT_ROOT, load_config
from kpibench.data.azure_calls import extract_summary_text, extract_transcript_text


@pytest.fixture(scope="module")
def cfg():
    return load_config(PROJECT_ROOT / "config" / "benchmark.yaml")


def test_every_model_has_nonzero_pricing_and_estimates_are_flagged(cfg):
    for mid, spec in cfg.model_registry.items():
        assert spec.pricing.input > 0 and spec.pricing.output > 0, mid
        if spec.pricing_is_estimate:
            assert spec.pricing_source.startswith("ESTIMATIVA"), mid


def test_estimates_are_commented_in_yaml():
    import re

    text = (PROJECT_ROOT / "config" / "models.yaml").read_text(encoding="utf-8")
    blocks = re.split(r"\n(?=  [a-z0-9][\w.\-]*:\n)", text)   # um bloco por modelo (chave com 2 espaços)
    flagged = [b for b in blocks if re.search(r"^    pricing_is_estimate: true", b, re.M)]
    assert len(flagged) >= 10
    for b in flagged:
        assert "# ESTIMATIVA" in b, b.splitlines()[0]


@pytest.mark.parametrize("payload,expected", [
    ("Cliente pediu segunda via da fatura e recebeu por SMS.", "Cliente pediu segunda via da fatura e recebeu por SMS."),
    ({"summary": "Cliente pediu segunda via da fatura e recebeu por SMS."}, "Cliente pediu segunda via"),
    ({"summary_items": [{"text": "Cliente contestou cobrança indevida."}, {"text": "Estorno lançado."}]},
     "Cliente contestou cobrança indevida.\nEstorno lançado."),
    ('{"resumo": {"content": "Cliente quer cancelar o plano por preço alto."}}', "Cliente quer cancelar"),
    ({"meta": {"url": "https://x/y.json"}, "body": {"x": "Texto longo do resumo da ligação de suporte técnico."}},
     "Texto longo do resumo"),
])
def test_extract_summary_text_shapes(payload, expected):
    assert extract_summary_text(payload).startswith(expected)


def test_extract_transcript_text_turns():
    payload = {"transcription_items": [{"speaker": "agent", "text": "Bom dia"}, {"role": "customer", "content": "Oi"}]}
    assert extract_transcript_text(payload) == "AGENT: Bom dia\nCUSTOMER: Oi"


def test_production_mapping_only_comparable_kpis():
    calls = pd.DataFrame({"call_id": ["a", "b", "c"], "prod_csat": [9.0, 5.0, np.nan],
                          "prod_solicitacao_resolvida": ["Sim", "Não", "Sim"], "prod_empatia": ["sim", "nao", "sim"],
                          "prod_odc": ["sim", "sim", "nao"]})
    prod, notes = live.map_production_baseline(calls)
    assert list(prod["prod_csat"].fillna(-1)) == [5.0, 3.0, -1]
    assert list(prod["prod_solicitacao_resolvida"]) == ["sim", "nao", "sim"]
    assert "prod_empatia" not in prod and "prod_odc" not in prod
    assert set(notes.loc[~notes["comparavel"], "kpi"]) >= {"empatia", "odc", "qualidade_resumo"}


def test_live_flow_simulated(tmp_path):
    c = live.build_config(models=["claude-haiku-5-5", "gpt-6-luna", "claude-opus-5-5"], judge_model="gpt-4o-mini",
                          simulated=True, n_bootstrap=100)
    c.run.output_dir = str(tmp_path)
    calls = live.sample_calls(20)
    est = live.estimate_cost(c, calls, "claude-opus-5-5")
    assert est.iloc[-1]["modelo"] == "TOTAL" and est.iloc[-1]["usd_estimado"] > 0
    lr = live.run_live(c, calls, "claude-opus-5-5")
    assert "claude-opus-5-5" not in c.models                    # referência sai dos candidatos
    res = live.load_results(lr.run_dir)
    board = live.model_scoreboard(res, c)
    assert set(board["model"]) == {"claude-haiku-5-5", "gpt-6-luna"}
    assert board["indice"].between(0, 100).all() and (board["usd_1k_ligacoes"] > 0).all()
    assert bool(board.set_index("model").at["gpt-6-luna", "preco_estimado"])
    assert not live.production_comparison(res).empty
    cid = live.most_divergent_calls(res, lr.dataset, 1)[0]
    dd = live.call_drilldown(res, lr.dataset, cid)
    assert {"Referência", "Produção", "gpt-6-luna"} <= set(dd.columns)
    assert "Mais barato" in live.executive_summary(board, res, lr, len(lr.dataset), "x") or "Nenhum" in \
        live.executive_summary(board, res, lr, len(lr.dataset), "x")


def test_charts_render_without_plotly(tmp_path):
    import sys

    from kpibench.reporting import aa_charts as A

    c = live.build_config(models=["claude-haiku-5-5", "gpt-6-luna"], simulated=True, n_bootstrap=100)
    c.run.output_dir = str(tmp_path)
    lr = live.run_live(c, live.sample_calls(15), "claude-opus-5-5")
    res = live.load_results(lr.run_dir)
    b = live.model_scoreboard(res, c)
    figs = [A.scorecards(b, res.routing, 15), A.leaderboard(b), A.quality_vs_cost(b), A.cost_bars(b),
            A.latency_chart(b, sla_s=5), A.reliability_bars(b), A.kpi_heatmap(res.metrics, res.routing),
            A.kpi_small_multiples(res.metrics, res.routing), A.production_dumbbell(live.production_comparison(res)),
            A.calls_per_day(lr.dataset)]
    for i, f in enumerate(figs):
        if f is not None:
            f.savefig(tmp_path / f"{i}.png")
    assert "plotly" not in sys.modules
