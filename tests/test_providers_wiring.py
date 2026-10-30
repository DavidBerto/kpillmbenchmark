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


def test_azure_reasoning_model_request(cfg, monkeypatch):
    from kpibench.llm.providers.openai_provider import OpenAIChatClient

    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://x.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "k")
    cli = OpenAIChatClient(cfg.model("gpt-5.4-mini"))
    seen = {}

    def fake_create(**kw):
        seen.update(kw)
        usage = SimpleNamespace(prompt_tokens=500, completion_tokens=300,
                                prompt_tokens_details=SimpleNamespace(cached_tokens=100),
                                completion_tokens_details=SimpleNamespace(reasoning_tokens=200))
        msg = SimpleNamespace(content='{"csat": 4}')
        return SimpleNamespace(usage=usage, choices=[SimpleNamespace(message=msg, finish_reason="stop")])

    monkeypatch.setattr(cli.client.chat.completions, "create", fake_create)
    r = cli.complete(LLMRequest(system="S", user="U"))
    assert "temperature" not in seen                       # reasoning model
    assert seen["max_completion_tokens"] == 1200
    assert seen["reasoning_effort"] == "low"
    assert seen["response_format"] == {"type": "json_object"}
    assert (r.usage.input_tokens, r.usage.cached_tokens, r.usage.output_tokens) == (400, 100, 300)


def test_anthropic_request(cfg, monkeypatch):
    from kpibench.llm.providers.anthropic_provider import AnthropicClient

    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    cli = AnthropicClient(cfg.model("claude-haiku-5-5"))
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
