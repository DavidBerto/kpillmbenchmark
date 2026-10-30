"""Cálculo de custo por chamada (medido) e por chamada em produção (projetado)."""
from __future__ import annotations

from kpibench.config import ModelSpec
from kpibench.llm.base import Usage

M = 1_000_000


def cost_usd(spec: ModelSpec, usage: Usage, batch: bool = False) -> float:
    """Custo medido de uma chamada a partir do usage normalizado."""
    p = spec.pricing
    c = (usage.input_tokens * p.input
         + usage.cached_tokens * p.cached_input
         + usage.cache_write_tokens * p.input * p.cache_write_multiplier
         + usage.output_tokens * p.output) / M
    if batch:
        c *= 1 - spec.batch_discount
    return c


def projected_cost_per_call(spec: ModelSpec, mean_input_tokens: float, mean_output_tokens: float,
                            static_prefix_tokens: float, cache_hit_ratio: float, batch: bool) -> float:
    """Custo esperado por chamada em regime de produção.

    O prefixo estático (system prompt) só é cacheável se tiver ao menos
    `min_cacheable_tokens`; nesse caso uma fração `cache_hit_ratio` é lida do cache.
    Escritas de cache são amortizadas (desprezíveis em volume alto) e ignoradas.
    """
    p = spec.pricing
    cacheable = static_prefix_tokens if static_prefix_tokens >= spec.min_cacheable_tokens else 0.0
    cached = cacheable * cache_hit_ratio
    full = max(mean_input_tokens - cached, 0.0)
    c = (full * p.input + cached * p.cached_input + mean_output_tokens * p.output) / M
    if batch:
        c *= 1 - spec.batch_discount
    return c
