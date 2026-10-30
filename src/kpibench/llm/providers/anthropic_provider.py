"""Anthropic Messages API."""
from __future__ import annotations

import time

from kpibench.config import ModelSpec
from kpibench.llm.base import LLMClient, LLMRequest, LLMResponse, TransientLLMError, Usage
from kpibench.llm.tokens import estimate_tokens


class AnthropicClient(LLMClient):
    def __init__(self, spec: ModelSpec):
        super().__init__(spec)
        try:
            import anthropic  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise ImportError("instale os provedores: pip install -e '.[llm]'") from e
        self._anthropic = anthropic
        self.client = anthropic.Anthropic()

    def complete(self, req: LLMRequest) -> LLMResponse:
        params = self.generation_params()
        system_block: dict = {"type": "text", "text": req.system}
        # cache_control só compensa se o prefixo estático atinge o mínimo cacheável
        if estimate_tokens(req.system) >= self.spec.min_cacheable_tokens:
            system_block["cache_control"] = {"type": "ephemeral"}
        kwargs = {
            "model": self.spec.model,
            "system": [system_block],
            "messages": [{"role": "user", "content": req.user}],
            "max_tokens": params.pop("max_tokens", 1024),
        }
        params.pop("reasoning_effort", None)
        kwargs.update(params)

        a = self._anthropic
        t0 = time.perf_counter()
        try:
            resp = self.client.messages.create(**kwargs)
        except (a.RateLimitError, a.APITimeoutError, a.APIConnectionError, a.InternalServerError) as e:
            raise TransientLLMError(str(e)) from e
        except a.APIStatusError as e:  # 529 overloaded
            if getattr(e, "status_code", None) in (429, 500, 502, 503, 529):
                raise TransientLLMError(str(e)) from e
            raise
        latency = time.perf_counter() - t0

        u = resp.usage
        usage = Usage(input_tokens=u.input_tokens or 0,
                      cached_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
                      cache_write_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
                      output_tokens=u.output_tokens or 0)
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        return LLMResponse(text=text, usage=usage, latency_s=latency, model=self.spec.model,
                           finish_reason=resp.stop_reason)
