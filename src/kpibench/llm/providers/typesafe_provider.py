"""Modelos de DECISÃO tipada (não geram texto): TypeSafe Jev e ScaleDown CDM (Decisions API).

As duas APIs têm o mesmo contrato de perguntas tipadas:
  corpo:    {"model": "...", "state": <texto | objeto>,
             "questions": {"<id>": {"type": "choice", "instructions": "...", "criteria": {"<opção>": "<descrição>"}},
                           "<id>": {"type": "score",  "instructions": "...", "criteria": ["<nível 0>", ...]},
                           "<id>": {"type": "noul",   "instructions": "..."}}}
  resposta: {"model": "...", "answers": {"<id>": {"type": "choice", "choice": "...", "probabilities": {...},
             "confidence": 0.85} | {"type": "score", "score": 1.7, "confidence": ...} | {"type": "noul", "noul": 0.98}},
             "usage": {"input_tokens": 650, "output_tokens": 70[, "cost": 0.0000325]}}

  TypeSafe Jev (docs.typesafe.ai/api)    POST https://api.typesafe.ai/v1/systemone · Authorization: Bearer <key>
      state livre (texto/objeto); textos acima de ~32 KB exigem o header de contexto longo.
      Via OpenRouter: base_url https://openrouter.ai/api, path /alpha/decisions, modelo typesafe/jev-1.14.
  ScaleDown CDM (docs.scaledown.ai/api-reference/decisions)
                                          POST https://api.scaledown.xyz/v1/scaledown · x-api-key: <key>
      modelo classify-1 (fixo) ou classify; state = string ou {"text": ...}; score com 2 a 10 níveis; cada
      pergunta é uma chamada cobrada à parte; usage.cost traz o custo em US$.

O payload chega pronto em `LLMRequest.user` (JSON) — o extrator monta as perguntas; este cliente só faz o
transporte, injeta o modelo, mede a latência e normaliza o `usage`. A resposta bruta volta em `text`.
"""
from __future__ import annotations

import json
import os
import socket
import time
import urllib.error
import urllib.request

from kpibench.config import ModelSpec
from kpibench.llm.base import LLMClient, LLMRequest, LLMResponse, TransientLLMError, Usage

TRANSIENT_STATUS = {408, 429, 500, 502, 503, 504, 529}
LONG_CONTEXT_BYTES = 32_000
LONG_CONTEXT_HEADER = "TypeSafe-Danger-Zone-Long-Context"


class DecisionAPIClient(LLMClient):
    default_base_url = ""
    default_path = ""
    default_key_env = ""
    auth = "bearer"                     # bearer | x-api-key
    vendor = ""

    def __init__(self, spec: ModelSpec):
        super().__init__(spec)
        env = spec.api_key_env or self.default_key_env
        self.api_key = os.environ.get(env)
        if not self.api_key:
            raise RuntimeError(f"defina a variável de ambiente {env} com a API key do {self.vendor}")
        p = spec.params
        self.url = (spec.base_url or self.default_base_url).rstrip("/") + p.get("path", self.default_path)
        self.timeout = float(p.get("timeout_s", 30))
        self.long_context = p.get("long_context", False)

    def generation_params(self) -> dict:
        # nada disto é enviado como parâmetro de geração; entra só na chave de cache
        return {k: v for k, v in self.spec.params.items() if k in ("path", "long_context")}

    def _headers(self, state_bytes: int) -> dict:
        h = {"Content-Type": "application/json"}
        if self.auth == "x-api-key":
            h["x-api-key"] = self.api_key
        else:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def complete(self, req: LLMRequest) -> LLMResponse:
        payload = json.loads(req.user)
        payload["model"] = self.spec.model
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        state_bytes = len(json.dumps(payload.get("state", ""), ensure_ascii=False).encode("utf-8"))
        request = urllib.request.Request(self.url, data=body, headers=self._headers(state_bytes), method="POST")
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:300]
            if e.code in TRANSIENT_STATUS:
                raise TransientLLMError(f"HTTP {e.code}: {detail}") from e
            raise RuntimeError(f"{self.vendor} HTTP {e.code}: {detail}") from e
        except (urllib.error.URLError, socket.timeout, TimeoutError) as e:
            raise TransientLLMError(f"rede/timeout: {e}") from e
        latency = time.perf_counter() - t0
        obj = json.loads(raw)
        u = obj.get("usage") or {}
        usage = Usage(input_tokens=int(u.get("input_tokens") or u.get("prompt_tokens") or 0),
                      output_tokens=int(u.get("output_tokens") or u.get("completion_tokens") or 0))
        cost = u.get("cost")
        return LLMResponse(text=raw, usage=usage, latency_s=latency, model=str(obj.get("model", self.spec.model)),
                           finish_reason="stop", reported_cost_usd=float(cost) if isinstance(cost, (int, float)) else None)


class TypeSafeJevClient(DecisionAPIClient):
    default_base_url = "https://api.typesafe.ai"
    default_path = "/v1/systemone"
    default_key_env = "TYPESAFE_API_KEY"
    auth = "bearer"
    vendor = "Jev (TypeSafe)"

    def __init__(self, spec: ModelSpec):
        super().__init__(spec)
        self.long_context = spec.params.get("long_context", "auto")   # auto | true | false

    def _headers(self, state_bytes: int) -> dict:
        h = super()._headers(state_bytes)
        if self.long_context is True or (self.long_context == "auto" and state_bytes > LONG_CONTEXT_BYTES):
            h[LONG_CONTEXT_HEADER] = "true"
        return h


class ScaleDownDecisionsClient(DecisionAPIClient):
    default_base_url = "https://api.scaledown.xyz"
    default_path = "/v1/scaledown"
    default_key_env = "SCALEDOWN_API_KEY"
    auth = "x-api-key"
    vendor = "ScaleDown"
