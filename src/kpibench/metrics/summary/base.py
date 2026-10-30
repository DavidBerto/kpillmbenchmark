"""Estrutura comum das métricas de qualidade de resumo."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MetricResult:
    name: str
    family: str            # lexical | factual_heuristica | embedding | nli | llm_juiz
    group: str             # eficiente | moderna
    reference: str         # referencia | fonte | pseudo_referencia(fonte) | sem_referencia
    values: np.ndarray     # um valor por registro (NaN = não calculado)
    direction: int = 1     # +1: maior é melhor; -1: menor é melhor
    compute: str = "CPU"   # CPU | GPU/CPU | API
    ms_per_item: float = float("nan")
    usd_per_1k: float = 0.0
    note: str = ""
    skipped_reason: str | None = None
    extra: dict = field(default_factory=dict)

    @property
    def oriented(self) -> np.ndarray:
        return self.values * self.direction


def skipped(name: str, family: str, group: str, reference: str, n: int, reason: str,
            compute: str = "GPU/CPU") -> MetricResult:
    return MetricResult(name, family, group, reference, np.full(n, np.nan), compute=compute,
                        skipped_reason=reason)
