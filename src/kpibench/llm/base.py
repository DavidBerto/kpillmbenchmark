"""Contrato comum dos clientes LLM."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any

from kpibench.config import ModelSpec


class TransientLLMError(RuntimeError):
    """Erro recuperável (rate limit, timeout, 5xx, overload) -> retry com backoff."""


@dataclass
class Usage:
    input_tokens: int = 0          # entrada cobrada a preço cheio (sem cache)
    cached_tokens: int = 0         # entrada lida do cache
    cache_write_tokens: int = 0    # entrada escrita no cache (Anthropic)
    output_tokens: int = 0         # saída total, inclui tokens de raciocínio
    reasoning_tokens: int = 0      # subconjunto de output_tokens (informativo)

    @property
    def total_input(self) -> int:
        return self.input_tokens + self.cached_tokens + self.cache_write_tokens


@dataclass
class LLMRequest:
    system: str
    user: str
    # metadados NÃO enviados ao modelo; usados para cache-key e pelo provider mock
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    text: str
    usage: Usage
    latency_s: float
    model: str
    finish_reason: str | None = None
    simulated: bool = False

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "LLMResponse":
        d = dict(d)
        d["usage"] = Usage(**d["usage"])
        return cls(**d)


class LLMClient(ABC):
    def __init__(self, spec: ModelSpec):
        self.spec = spec

    @abstractmethod
    def complete(self, req: LLMRequest) -> LLMResponse:  # pragma: no cover - interface
        ...

    def generation_params(self) -> dict[str, Any]:
        """Parâmetros efetivos (entram na chave de cache)."""
        p = dict(self.spec.params)
        if not self.spec.supports_temperature:
            p.pop("temperature", None)
        return p
