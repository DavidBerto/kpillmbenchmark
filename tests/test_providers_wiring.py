"""Verifica a montagem das requisições reais sem rede (SDK com create() substituído)."""
from types import SimpleNamespace

import pytest

from kpibench.config import PROJECT_ROOT, load_config
from kpibench.llm.base import LLMRequest

openai = pytest.importorskip("openai")
anthropic = pytest.importorskip("anthropic")


@pytest.fixture(scope="module")
def cfg():
    return load_config(PROJECT_ROOT / "config" / "benchmark.yaml")


def _fake_openai_create(seen: dict, content: str = '{"csat": 4}'):
    def fake_create(**kw):
        seen.update(kw)
        usage = SimpleNamespace(prompt_tokens=500, completion_tokens=300,
                                prompt_tokens_details=SimpleNamespace(cached_tokens=100),
                                completion_tokens_details=SimpleNamespace(reasoning_tokens=200))
        msg = SimpleNamespace(content=content)
        return SimpleNamespace(usage=usage, choices=[SimpleNamespace(message=msg, finish_reason="stop")])
    return fake_create


def test_all_models_route_through_nexus(cfg):
    assert {s.provider for s in cfg.model_registry.values()} == {"nexus"}


def test_nexus_reasoning_model_request(cfg, monkeypatch):
    from kpibench.llm.providers.nexus_provider import NexusChatClient

    monkeypatch.setenv("base_url", "https://nexus.exemplo/v1")
    monkeypatch.setenv("bearer_token", "t")
    cli = NexusChatClient(cfg.model("gpt-6-luna"))
    seen = {}
    monkeypatch.setattr(cli.client.chat.completions, "create", _fake_openai_create(seen))
    r = cli.complete(LLMRequest(system="S", user="U"))
    assert seen["model"] == "gpt-6-luna"
    assert "temperature" not in seen                       # modelo de raciocínio
    assert seen["max_tokens"] == 1200 and seen["reasoning_effort"] == "low"
    assert seen["response_format"] == {"type": "json_object"}
    assert (r.usage.input_tokens, r.usage.cached_tokens, r.usage.output_tokens) == (400, 100, 300)


def test_nexus_claude_request(cfg, monkeypatch):
    from kpibench.llm.providers.nexus_provider import NexusChatClient

    monkeypatch.setenv("base_url", "https://nexus.exemplo/v1")
    monkeypatch.setenv("bearer_token", "t")
    cli = NexusChatClient(cfg.model("claude-haiku-4-5"))
    seen = {}
    monkeypatch.setattr(cli.client.chat.completions, "create", _fake_openai_create(seen))
    cli.complete(LLMRequest(system="S", user="U"))
    assert seen["temperature"] == 0.0 and seen["max_tokens"] == 800
    assert "response_format" not in seen and "reasoning_effort" not in seen


def test_nexus_claude_5_5_family_drops_temperature(cfg, monkeypatch):
    """claude-{haiku,sonnet,opus}-5-5: o gateway Nexus rejeita temperature != 1
    (BadRequestError real, visto em outputs/teste_ligacoes_20261030_083719), por
    isso supports_temperature: false em config/models.yaml para os três."""
    from kpibench.llm.providers.nexus_provider import NexusChatClient

    monkeypatch.setenv("base_url", "https://nexus.exemplo/v1")
    monkeypatch.setenv("bearer_token", "t")
    for model_id in ("claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"):
        cli = NexusChatClient(cfg.model(model_id))
        seen = {}
        monkeypatch.setattr(cli.client.chat.completions, "create", _fake_openai_create(seen))
        cli.complete(LLMRequest(system="S", user="U"))
        assert "temperature" not in seen, f"{model_id} não deveria enviar temperature"


def test_anthropic_request(cfg, monkeypatch):
    from kpibench.llm.providers.anthropic_provider import AnthropicClient

    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    cli = AnthropicClient(cfg.model("claude-haiku-4-5"))
    seen = {}

    def fake_create(**kw):
        seen.update(kw)
        usage = SimpleNamespace(input_tokens=450, output_tokens=60, cache_read_input_tokens=0,
                                cache_creation_input_tokens=0)
        return SimpleNamespace(usage=usage, content=[SimpleNamespace(type="text", text='{"csat": 5}')],
                               stop_reason="end_turn")

    monkeypatch.setattr(cli.client.messages, "create", fake_create)
    r = cli.complete(LLMRequest(system="S", user="U"))
    assert seen["temperature"] == 0.0 and seen["max_tokens"] == 800
    assert "cache_control" not in seen["system"][0]        # prefixo curto: sem cache
    assert r.text == '{"csat": 5}' and r.usage.input_tokens == 450
