"""Gateway Nexus: 1 endpoint OpenAI-compatível (base_url + bearer_token) que expõe
vários provedores (OpenAI, Anthropic, Google, Grok, DeepSeek, Kimi etc.) sob a
mesma API de chat completions.

Validado em notebooks/connection_test.ipynb (célula "Conexão com as LLMs via
Nexus"). Variáveis de ambiente usadas (mesmos nomes do notebook, ver .env.example):
  base_url       endpoint do gateway, ex.: https://nexus.empresa.com/v1
  bearer_token   token de autenticação (enviado como "Authorization: Bearer ...")
"""
from __future__ import annotations

import os
import time

from kpibench.config import ModelSpec
from kpibench.llm.base import LLMClient, LLMRequest, LLMResponse, TransientLLMError, Usage


class NexusChatClient(LLMClient):
    def __init__(self, spec: ModelSpec):
        super().__init__(spec)
        try:
            import openai  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise ImportError("instale os provedores: pip install -e '.[llm]'") from e
        self._openai = openai
        try:
            base_url = os.environ["base_url"]
            bearer_token = os.environ["bearer_token"]
        except KeyError as e:
            raise RuntimeError(
                "defina 'base_url' e 'bearer_token' no ambiente (.env) para usar o provider 'nexus'"
            ) from e
        self.client = openai.OpenAI(base_url=base_url, api_key=bearer_token)

    def complete(self, req: LLMRequest) -> LLMResponse:
        params = self.generation_params()
        kwargs: dict = {
            "model": self.spec.model,
            "messages": [{"role": "system", "content": req.system},
                         {"role": "user", "content": req.user}],
        }
        max_tokens = params.pop("max_tokens", None)
        if max_tokens:
            # o gateway expõe a API padrão de chat completions para todos os
            # modelos roteados (inclusive os não-OpenAI) -> sempre "max_tokens".
            kwargs["max_tokens"] = max_tokens
        if "reasoning_effort" in params:
            kwargs["reasoning_effort"] = params.pop("reasoning_effort")
        kwargs.update(params)
        if self.spec.json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        o = self._openai
        t0 = time.perf_counter()
        try:
            resp = self.client.chat.completions.create(**kwargs)
        except (o.RateLimitError, o.APITimeoutError, o.APIConnectionError, o.InternalServerError) as e:
            raise TransientLLMError(str(e)) from e
        latency = time.perf_counter() - t0

        u = resp.usage
        cached = getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", 0) or 0
        reasoning = getattr(getattr(u, "completion_tokens_details", None), "reasoning_tokens", 0) or 0
        usage = Usage(input_tokens=(u.prompt_tokens or 0) - cached, cached_tokens=cached,
                      output_tokens=u.completion_tokens or 0, reasoning_tokens=reasoning)
        choice = resp.choices[0]
        return LLMResponse(text=choice.message.content or "", usage=usage, latency_s=latency,
                           model=self.spec.model, finish_reason=choice.finish_reason)
