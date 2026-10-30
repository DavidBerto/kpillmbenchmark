"""Estimativa de tokens (usada no `estimate` e no provider mock).

Rodadas reais usam o `usage` devolvido pela API. Sem tiktoken, usa a heurística
de ~3.6 caracteres/token para português (tokenizers o200k/cl100k ficam entre 3.3 e 4.0).
Tokenizers da Anthropic e Google diferem: trate a estimativa como ordem de grandeza.
"""
from __future__ import annotations

import math
from functools import lru_cache

CHARS_PER_TOKEN_PT = 3.6


@lru_cache(maxsize=1)
def _encoder():
    try:
        import tiktoken  # type: ignore

        return tiktoken.get_encoding("o200k_base")
    except Exception:  # noqa: BLE001 - dependência opcional
        return None


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    enc = _encoder()
    if enc is not None:
        return len(enc.encode(text))
    return math.ceil(len(text) / CHARS_PER_TOKEN_PT)
