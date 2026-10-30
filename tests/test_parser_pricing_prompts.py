import pytest

from kpibench.config import PROJECT_ROOT, load_config
from kpibench.extraction.parser import extract_json, parse_claims, parse_judge_score, parse_kpi_output
from kpibench.llm.base import Usage
from kpibench.llm.pricing import cost_usd, projected_cost_per_call
from kpibench.prompts.renderer import PromptRenderer


@pytest.fixture(scope="module")
def cfg():
    return load_config(PROJECT_ROOT / "config" / "benchmark.yaml")


# ----------------------------------------------------------------- parser
def test_extract_json_envelopes():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Claro! Segue: {"a": {"b": "x}"}} fim') == {"a": {"b": "x}"}}
    assert extract_json('{"a": 1') is None
    assert extract_json("sem json") is None


def test_parse_kpi_domains(cfg):
    k = cfg.kpi_registry
    assert parse_kpi_output('{"csat": 4}', k["csat"]).value == 4
    assert not parse_kpi_output('{"csat": 7}', k["csat"]).valid
    assert parse_kpi_output('{"resolvida": "Não"}', k["solicitacao_resolvida"]).value == "nao"
    bad = parse_kpi_output('{"resolvida": "talvez"}', k["solicitacao_resolvida"])
    assert not bad.valid and bad.error.startswith("rotulo_invalido")
    assert parse_kpi_output('{"odc": "Protocolo Aberto"}', k["odc"]).value == "protocolo_aberto"
    assert not parse_kpi_output('{"outra": 1}', k["csat"]).valid
    assert parse_kpi_output('{"razao_csat": "  ok  "}', k["razao_csat"]).value == "ok"


def test_judge_and_claims_parsers():
    assert parse_judge_score('{"nota": 5}') == 1.0
    assert parse_judge_score('{"nota": 3}') == 0.5
    assert parse_judge_score('{"nota": 9}') is None
    frac, n, contra = parse_claims('{"claims": [{"veredito": "suportada"}, {"veredito": "contradita"}]}')
    assert (frac, n, contra) == (0.5, 2, 1)


# ----------------------------------------------------------------- custo
def test_cost_and_projection(cfg):
    spec = cfg.model("claude-sonnet-5-5")  # 2.00 / 0.10 / 10.00 por 1M
    u = Usage(input_tokens=1_000_000, cached_tokens=1_000_000, output_tokens=1_000_000)
    assert cost_usd(spec, u) == pytest.approx(2.0 + 0.10 + 10.0)
    assert cost_usd(spec, u, batch=True) == pytest.approx((2.0 + 0.10 + 10.0) * 0.5)
    # prefixo abaixo do mínimo cacheável: cache não se aplica
    a = projected_cost_per_call(spec, 1000, 100, static_prefix_tokens=500, cache_hit_ratio=1.0, batch=False)
    assert a == pytest.approx((1000 * 2.0 + 100 * 10.0) / 1e6)
    # prefixo cacheável: 2000 de 3000 tokens lidos do cache
    b = projected_cost_per_call(spec, 3000, 100, static_prefix_tokens=2000, cache_hit_ratio=1.0, batch=False)
    assert b == pytest.approx((1000 * 2.0 + 2000 * 0.10 + 100 * 10.0) / 1e6)


# ----------------------------------------------------------------- prompts
def test_every_kpi_prompt_renders_and_declares_its_field(cfg):
    r = PromptRenderer(PROJECT_ROOT / "prompts")
    for k in cfg.kpi_registry.values():
        p = r.render(f"kpis/{k.prompt}", resumo="Resumo X.", transcricao="ATENDENTE: oi", has_transcript=True,
                     labels=k.labels or {}, scale=k.scale.model_dump() if k.scale else {})
        assert "Resumo X." in p.user and "Resumo X." not in p.system  # parte variável fora do prefixo
        assert f'"{k.output_field}"' in p.system, k.name
        if k.labels:
            assert all(f'"{lab}"' in p.system for lab in k.labels), k.name
        assert len(p.template_hash) == 12


def test_prompt_is_model_independent(cfg):
    """O render não recebe nenhuma informação do modelo: o mesmo texto vai para todos."""
    r = PromptRenderer(PROJECT_ROOT / "prompts")
    k = cfg.kpi_registry["csat"]
    a = r.render("kpis/csat.md", resumo="R", transcricao="", has_transcript=False, labels={}, scale=k.scale.model_dump())
    b = r.render("kpis/csat.md", resumo="R", transcricao="", has_transcript=False, labels={}, scale=k.scale.model_dump())
    assert (a.system, a.user, a.template_hash) == (b.system, b.user, b.template_hash)
