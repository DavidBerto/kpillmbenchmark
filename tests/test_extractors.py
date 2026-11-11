"""Backends de produção (LLM por KPI, LLM consolidado, CDM, Jev, híbrido) e o comparativo do notebook 05."""
import io
import json
import urllib.error
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from kpibench import backend_comparison as bc
from kpibench import extractors as X
from kpibench import live
from kpibench import single_call as sc
from kpibench.config import PROJECT_ROOT, load_config
from kpibench.llm.base import LLMRequest, TransientLLMError


@pytest.fixture(scope="module")
def cfg():
    c = load_config(PROJECT_ROOT / "config" / "benchmark.yaml")
    c.run.cache_path = None
    return c


def _walk_objects(schema):
    if isinstance(schema, dict):
        if schema.get("type") == "object":
            yield schema
        for v in schema.values():
            yield from _walk_objects(v)


def test_cdm_schema_is_strict_and_encodes_kpi_domains(cfg):
    kpis = sc.single_call_kpis(cfg, True)
    schema = X.kpi_json_schema(kpis)
    for obj in _walk_objects(schema):        # exigências do modo strict
        assert obj["additionalProperties"] is False
        assert set(obj["required"]) == set(obj["properties"])
    props = schema["properties"]["kpis"]["properties"]
    assert set(props) == {k.name for k in kpis}
    csat = cfg.kpi_registry["csat"]
    assert props["csat"]["properties"]["valor"]["enum"] == list(range(csat.scale.min, csat.scale.max + 1))
    assert props["odc"]["properties"]["valor"]["enum"] == list(cfg.kpi_registry["odc"].labels)
    assert props["solver_reasoning"]["properties"]["valor"] == {"type": "string"}
    assert "justificativa" not in X.kpi_json_schema(kpis, justificativas=False)["properties"]["kpis"]["properties"]["csat"]["properties"]


def test_jev_payload_types_and_coverage(cfg):
    ctx = X.ExtractorContext(cfg)
    kpis = sc.single_call_kpis(cfg, True)
    jev = X.build_extractor("jev", ctx, kpis)
    assert {k.name for k in jev.supported_kpis} == {k.name for k in kpis if k.type in ("categorical", "ordinal")}
    p = jev.payload({"call_id": "1", "transcricao": "ATENDENTE: oi", "resumo": "r"}, jev.supported_kpis)
    assert p["state"] == {"transcricao": "ATENDENTE: oi", "resumo": "r"}       # resumo só por qualidade_resumo
    for k in jev.supported_kpis:
        q = p["questions"][k.name]
        if k.type == "categorical":
            assert q["type"] == "choice" and q["criteria"] == dict(k.labels)
        else:
            assert q["type"] == "score" and len(q["criteria"]) == k.scale.max - k.scale.min + 1
        assert "transcrição" in q["instructions"]
    no_summary = jev.payload({"call_id": "1", "transcricao": "t"}, [cfg.kpi_registry["csat"]])
    assert "resumo" not in no_summary["state"]


def test_jev_answer_parsing(cfg):
    csat, res, emp = cfg.kpi_registry["csat"], cfg.kpi_registry["solicitacao_resolvida"], cfg.kpi_registry["empatia"]
    text = json.dumps({"model": "jev-1.13.0", "answers": {
        "csat": {"type": "score", "score": 2.6, "confidence": 0.7},                # índice 2.6 -> nível 3 -> nota 4
        "solicitacao_resolvida": {"type": "choice", "choice": "Parcial ", "confidence": 0.4},
        "empatia": {"type": "choice", "choice": "talvez"}}, "usage": {"input_tokens": 100}})
    v = X.JevExtractor.parse_answers(text, [csat, res, emp, cfg.kpi_registry["odc"]])
    assert v["csat"].value == 4 and v["csat"].valid and v["csat"].confidence == 0.7
    assert v["solicitacao_resolvida"].value == "parcial" and v["solicitacao_resolvida"].confidence == 0.4
    assert not v["empatia"].valid and v["empatia"].error.startswith("rotulo_invalido")
    assert v["odc"].error == "campo_ausente:odc"
    assert all(x.error == "json_invalido" for x in X.JevExtractor.parse_answers("erro", [csat]).values())


def _fake_urlopen(seen, body: dict, status: int = 200):
    def urlopen(request, timeout=None):
        seen["url"], seen["headers"] = request.full_url, dict(request.header_items())
        seen["payload"], seen["timeout"] = json.loads(request.data), timeout
        if status != 200:
            raise urllib.error.HTTPError(request.full_url, status, "err", {}, io.BytesIO(b'{"detail":"x"}'))
        return io.BytesIO(json.dumps(body).encode())
    return urlopen


def test_typesafe_client_request_and_errors(cfg, monkeypatch):
    from kpibench.llm.providers import typesafe_provider as tp

    monkeypatch.setenv("TYPESAFE_API_KEY", "segredo")
    ctx = X.ExtractorContext(cfg)
    jev = X.build_extractor("jev", ctx, [cfg.kpi_registry["csat"]])
    cli = tp.TypeSafeJevClient(jev.spec)
    seen = {}
    monkeypatch.setattr(tp.urllib.request, "urlopen", _fake_urlopen(seen, {
        "model": "jev-1.13.0", "answers": {}, "usage": {"input_tokens": 321, "output_tokens": 9}}))
    resp = cli.complete(LLMRequest(system="", user=json.dumps({"state": {"transcricao": "x"}, "questions": {}})))
    assert seen["url"] == "https://api.typesafe.ai/v1/systemone"
    assert seen["headers"]["Authorization"] == "Bearer segredo"
    assert seen["payload"]["model"] == "jev-1.13.0"
    assert tp.LONG_CONTEXT_HEADER not in seen["headers"]
    assert resp.usage.input_tokens == 321 and resp.usage.output_tokens == 9
    cli.complete(LLMRequest(system="", user=json.dumps({"state": {"transcricao": "x" * 40_000}, "questions": {}})))
    assert seen["headers"][tp.LONG_CONTEXT_HEADER.capitalize()] == "true"      # urllib capitaliza o header
    monkeypatch.setattr(tp.urllib.request, "urlopen", _fake_urlopen(seen, {}, status=429))
    with pytest.raises(TransientLLMError):
        cli.complete(LLMRequest(system="", user='{"state": "x", "questions": {}}'))
    monkeypatch.setattr(tp.urllib.request, "urlopen", _fake_urlopen(seen, {}, status=401))
    with pytest.raises(RuntimeError, match="401"):
        cli.complete(LLMRequest(system="", user='{"state": "x", "questions": {}}'))
    monkeypatch.delenv("TYPESAFE_API_KEY")
    with pytest.raises(RuntimeError, match="TYPESAFE_API_KEY"):
        tp.TypeSafeJevClient(jev.spec)


def _fake_create(seen: dict, content: str, prompt_tokens: int = 900, completion_tokens: int = 80):
    def create(**kw):
        seen.update(kw)
        usage = SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                                prompt_tokens_details=None, completion_tokens_details=None)
        return SimpleNamespace(usage=usage, choices=[SimpleNamespace(message=SimpleNamespace(content=content),
                                                                     finish_reason="stop")])
    return create


@pytest.mark.parametrize("mode", ["json_schema", "guided_json"])
def test_structured_output_llm_sends_constrained_schema(cfg, monkeypatch, mode):
    pytest.importorskip("openai")
    monkeypatch.setenv("SO_BASE_URL", "https://llm.exemplo/v1")
    monkeypatch.setenv("SO_API_KEY", "k")
    conf = X.load_extractor_config(cfg)
    conf["llm_estruturado"] = {"type": "llm_json_schema", "provider": "openai_compatible", "model": "modelo-restrito",
                               "base_url_env": "SO_BASE_URL", "api_key_env": "SO_API_KEY", "schema_mode": mode,
                               "pricing": {"input": 0.10, "cached_input": 0.025, "output": 0.40}}
    ctx = X.ExtractorContext(cfg, conf=conf)
    kpis = [cfg.kpi_registry["csat"], cfg.kpi_registry["solver_reasoning"]]
    ex = X.build_extractor("llm_estruturado", ctx, kpis)
    cli = ctx.executor.client(ex.spec.id)
    assert str(cli.client.base_url).startswith("https://llm.exemplo/v1")
    seen = {}
    content = json.dumps({"kpis": {"csat": {"justificativa": "ok", "valor": 4}, "solver_reasoning": {"valor": "Estornou."}}})
    monkeypatch.setattr(cli.client.chat.completions, "create", _fake_create(seen, content))
    e = ex.extract({"call_id": "1", "transcricao": "ATENDENTE: oi", "resumo": ""})
    assert seen["model"] == "modelo-restrito" and seen["max_tokens"] == 4000
    if mode == "json_schema":
        rf = seen["response_format"]
        assert rf["type"] == "json_schema" and rf["json_schema"]["strict"] is True
        assert rf["json_schema"]["schema"] == X.kpi_json_schema(kpis)
    else:
        assert seen["extra_body"]["guided_json"] == X.kpi_json_schema(kpis)
    assert e.values["csat"].value == 4 and e.values["solver_reasoning"].valid
    assert e.cost_usd == pytest.approx((900 * 0.10 + 80 * 0.40) / 1e6)


def test_structured_output_llm_via_nexus_uses_models_yaml_pricing(cfg):
    ctx = X.ExtractorContext(cfg)
    ex = X.build_extractor("llm_estruturado", ctx, [cfg.kpi_registry["csat"]])
    assert ex.spec.id == "gpt-6-luna@llm_estruturado" and ex.spec.provider == "nexus" and not ex.spec.json_mode
    assert ex.spec.pricing == cfg.model("gpt-6-luna").pricing
    assert ex.request_params([cfg.kpi_registry["csat"]])["response_format"]["json_schema"]["strict"] is True


def test_scaledown_cdm_decisions_and_summarize(cfg, monkeypatch):
    """ScaleDown: Decisions API (x-api-key, classify-1, usage.cost) + Summarize OpenAI-compatível (summarize-1)."""
    pytest.importorskip("openai")
    from kpibench.llm.providers import typesafe_provider as tp

    monkeypatch.setenv("SCALEDOWN_API_KEY", "sd-key")
    ctx = X.ExtractorContext(cfg)
    csat, odc, why = cfg.kpi_registry["csat"], cfg.kpi_registry["odc"], cfg.kpi_registry["razao_csat"]
    cdm = X.build_extractor("cdm", ctx, [csat, odc, why])
    assert cdm.supports(why) and cdm.supports(odc)                       # cobre todos os KPIs
    seen = {}
    monkeypatch.setattr(tp.urllib.request, "urlopen", _fake_urlopen(seen, {
        "model": "classify-1",
        "answers": {"csat": {"type": "score", "score": 3.2, "confidence": 0.81},
                    "odc": {"type": "choice", "choice": "protocolo_aberto", "confidence": 0.66}},
        "usage": {"input_tokens": 2000, "output_tokens": 8, "cost": 0.0001}}))
    summ = ctx.executor.client(cdm.summ_spec.id)
    assert str(summ.client.base_url).startswith("https://api.scaledown.xyz/v1")
    seen_s = {}
    monkeypatch.setattr(summ.client.chat.completions, "create", _fake_create(
        seen_s, json.dumps({"summary": "Cliente satisfeito: estorno feito na ligação.", "finish_reason": "stop"}),
        prompt_tokens=700, completion_tokens=12))
    e = cdm.extract({"call_id": "9", "transcricao": "ATENDENTE: estornei o valor.", "resumo": ""})
    assert seen["url"] == "https://api.scaledown.xyz/v1/scaledown"
    assert seen["headers"]["X-api-key"] == "sd-key" and "Authorization" not in seen["headers"]
    assert seen["payload"]["model"] == "classify-1" and set(seen["payload"]["questions"]) == {"csat", "odc"}
    assert seen["payload"]["state"]["text"].startswith("TRANSCRIÇÃO:")
    assert seen_s["model"] == "summarize-1" and seen_s["max_tokens"] == 150
    assert "RAZÃO DO CSAT" in seen_s["messages"][0]["content"]          # rubrica do KPI como instrução
    assert e.values["csat"].value == 4 and e.values["csat"].confidence == 0.81
    assert e.values["odc"].value == "protocolo_aberto"
    assert e.values["razao_csat"].value.startswith("Cliente satisfeito") and e.values["razao_csat"].valid
    decisions, summarize = e.requests
    assert decisions.cost_usd == pytest.approx(0.0001)                    # usage.cost informado pela API
    assert summarize.cost_usd == pytest.approx(700 * 0.05 / 1e6)           # saída gratuita
    assert e.latency_s == pytest.approx(max(decisions.latency_s, summarize.latency_s))


@pytest.mark.parametrize("hybrid,primary", [("jev_hibrido", "jev"), ("cdm_hibrido", "cdm")])
def test_hybrid_routes_text_and_low_confidence_to_secondary(hybrid, primary):
    c = live.build_config(models=["gpt-6-luna"], simulated=True)
    calls = live.sample_calls(10).to_dict("records")
    kpis = sc.single_call_kpis(c, True)
    text = {k.name for k in kpis if k.type == "text"}
    for thr, expect_all in ((0.0, False), (1.01, True)):
        conf = X.load_extractor_config(c)
        conf[hybrid] = {**conf[hybrid], "confidence_threshold": thr}
        ctx = X.ExtractorContext(c, conf=conf)
        hyb = X.build_extractor(hybrid, ctx, kpis)
        assert hyb.kind == f"hybrid_{primary}"
        for e in X.run_extractor(hyb, calls, 2):
            assert set(e.values) == {k.name for k in kpis} and len(e.requests) == 2
            for name, v in e.values.items():
                if name in text:
                    assert v.source == "llm_consolidado"                 # texto livre sempre no LLM
                elif expect_all:
                    assert v.source == "llm_consolidado (fallback)" and v.confidence is not None
                else:
                    assert v.source == primary
            assert e.latency_s == pytest.approx(sum(r.latency_s for r in e.requests))


def test_backend_comparison_simulated_end_to_end(tmp_path):
    c = live.build_config(models=["gpt-6-luna"], judge_model="gpt-4o-mini", simulated=True, n_bootstrap=50)
    c.run.output_dir = str(tmp_path)
    calls = live.sample_calls(12)
    backends = ["llm_por_kpi", "llm_consolidado", "llm_estruturado", "cdm", "cdm_hibrido", "jev", "jev_hibrido"]
    est = bc.estimate_backends(c, calls, backends)
    assert list(est["backend"]) == backends and (est["usd_estimado"] > 0).all()
    st = bc.run_backend_comparison(c, calls, backends, sla_s=3.0)

    s = st.summary.set_index("backend")
    n_kpis = len(sc.single_call_kpis(c, True))
    assert list(s.index) == backends
    assert s.at["jev", "kpis_cobertos"] < n_kpis and s.at["jev_hibrido", "kpis_cobertos"] == n_kpis
    assert s.at["cdm", "kpis_cobertos"] == n_kpis and s.at["cdm", "tipo"] == "cdm"
    assert s.at["cdm_hibrido", "tipo"] == "hybrid_cdm" and s.at["jev_hibrido", "tipo"] == "hybrid_jev"
    assert s.at["llm_por_kpi", "requisicoes_por_ligacao"] == n_kpis and s.at["jev", "requisicoes_por_ligacao"] == 1
    assert set(st.common_kpis) == set(st.coverage["jev"])
    assert s["status_ce"].isin(["fronteira", "dominado", "dominado_estendido", "sem_dados"]).all()
    assert s.at["llm_por_kpi", "decisao_vs_base"] == "base"
    # o cálculo por componente reproduz o custo medido
    tot = st.breakdown.groupby("backend")["usd_1k_ligacoes"].sum()
    measured = st.calls.groupby("backend")["cost_usd"].mean() * 1000
    for b in backends:
        assert tot[b] == pytest.approx(measured[b], rel=1e-9)
    rec = st.recommendation
    assert rec["recomendado"] in backends and "jev" in rec["cobertura_parcial"]
    assert not st.calibration.empty and set(st.calibration["tipo_linha"]) == {"faixa", "limiar"}
    rep = bc.final_report(st)
    for sec in ("## 1. Qual compensa mais", "## 2. Custo", "## 3. Latência", "## 4. Qualidade",
                "## 5. Relação qualidade × custo", "## 6. Cálculo do custo", "RODADA SIMULADA"):
        assert sec in rep
    for b in backends:
        lines = st.jsonl_path(b).read_text(encoding="utf-8").splitlines()
        assert len(lines) == len(st.dataset) and "kpis" in json.loads(lines[0])

    from kpibench.reporting import backend_charts as bch

    figs = [bch.quality_cost_frontier(st.summary), bch.cost_per_quality(st.summary), bch.cost_bars(st.summary, 1000),
            bch.cost_breakdown_chart(st.breakdown), bch.tokens_chart(st.summary), bch.latency_chart(st.summary, 3.0),
            bch.quality_heatmap(st.quality, st.backends), bch.diff_forest(st.comparisons, st.summary, "llm_por_kpi"),
            bch.calibration_chart(st.calibration, 0.3)]
    for i, f in enumerate(figs):
        f.savefig(tmp_path / f"{i}.png")
    bch.style_summary(st.summary).to_html()


def test_cli_extract_simulated(tmp_path):
    from kpibench.cli import main

    inp = tmp_path / "ligacoes.csv"
    live.sample_calls(10)[["call_id", "transcricao", "resumo"]].to_csv(inp, index=False)
    out = tmp_path / "kpis.jsonl"
    assert main(["extract", "--backend", "jev", "--input", str(inp), "--output", str(out), "--simulated",
                 "--kpis", "csat,odc,solver_reasoning"]) == 0
    rows = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 10 and set(rows[0]["kpis"]) == {"csat", "odc"}     # Jev não cobre texto livre
    assert rows[0]["uso"]["custo_usd"] > 0 and rows[0]["kpis"]["csat"]["fonte"] == "jev"
