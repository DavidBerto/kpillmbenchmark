"""Azure OpenAI, OpenAI e qualquer endpoint OpenAI-compatível (Gemini, vLLM, Ollama)."""
from __future__ import annotations

import os
import time

from kpibench.config import ModelSpec
from kpibench.llm.base import LLMClient, LLMRequest, LLMResponse, TransientLLMError, Usage


class OpenAIChatClient(LLMClient):
    def __init__(self, spec: ModelSpec):
        super().__init__(spec)
        try:
            import openai  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise ImportError("instale os provedores: pip install -e '.[llm]'") from e
        self._openai = openai
        if spec.provider == "azure_openai":
            self.client = openai.AzureOpenAI(
                azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                api_key=os.environ["AZURE_OPENAI_API_KEY"],
                api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2025-04-01-preview"),
            )
        elif spec.provider == "openai":
            self.client = openai.OpenAI()
        else:  # openai_compatible
            key = os.environ.get(spec.api_key_env or "", "not-needed")
            self.client = openai.OpenAI(base_url=spec.base_url, api_key=key)

    def complete(self, req: LLMRequest) -> LLMResponse:
        params = self.generation_params()
        kwargs: dict = {
            "model": self.spec.model,
            "messages": [{"role": "system", "content": req.system},
                         {"role": "user", "content": req.user}],
        }
        max_tokens = params.pop("max_tokens", None)
        if max_tokens:
            # modelos de raciocínio exigem max_completion_tokens; endpoints compatíveis usam max_tokens
            key = "max_tokens" if self.spec.provider == "openai_compatible" else "max_completion_tokens"
            kwargs[key] = max_tokens
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
