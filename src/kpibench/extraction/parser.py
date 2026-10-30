"""Parsing robusto da saída JSON e normalização do valor do KPI.

Política: a extração é TOLERANTE no envelope (cercas ```json, texto em volta) e
ESTRITA no domínio (rótulo fora da taxonomia ou nota fora da escala = inválido).
Saídas inválidas não são descartadas: entram na avaliação como erro.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from kpibench.config import KPISpec
from kpibench.data.loader import normalize_label, to_ordinal

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S | re.I)


@dataclass
class ParsedOutput:
    value: Any
    valid: bool
    error: str | None = None
    extras: dict = field(default_factory=dict)


def extract_json(text: str) -> dict | None:
    if not text:
        return None
    candidates = [m.group(1) for m in _FENCE.finditer(text)] + [text]
    for c in candidates:
        c = c.strip()
        try:
            obj = json.loads(c)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        obj = _first_balanced_object(c)
        if obj is not None:
            return obj
    return None


def _first_balanced_object(s: str) -> dict | None:
    start = s.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(s)):
            ch = s[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(s[start:i + 1])
                        return obj if isinstance(obj, dict) else None
                    except json.JSONDecodeError:
                        break
        start = s.find("{", start + 1)
    return None


def parse_kpi_output(text: str, kpi: KPISpec) -> ParsedOutput:
    obj = extract_json(text)
    if obj is None:
        return ParsedOutput(None, False, "json_invalido")
    if kpi.output_field not in obj:
        return ParsedOutput(None, False, f"campo_ausente:{kpi.output_field}")
    raw = obj[kpi.output_field]
    extras = {k: v for k, v in obj.items() if k != kpi.output_field}

    if kpi.type == "ordinal":
        v = to_ordinal(raw, kpi)
        return ParsedOutput(v, v is not None, None if v is not None else f"fora_da_escala:{raw}", extras)
    if kpi.type == "categorical":
        v = normalize_label(raw)
        ok = v in (kpi.labels or {})
        return ParsedOutput(v if ok else None, ok, None if ok else f"rotulo_invalido:{raw}", extras)
    v = str(raw).strip() if raw is not None else ""
    return ParsedOutput(v or None, bool(v), None if v else "texto_vazio", extras)


def parse_judge_score(text: str, key: str = "nota", lo: int = 1, hi: int = 5) -> float | None:
    """Nota do juiz normalizada para [0, 1]."""
    obj = extract_json(text)
    if not obj or key not in obj:
        return None
    try:
        v = float(obj[key])
    except (TypeError, ValueError):
        return None
    if not lo <= v <= hi:
        return None
    return (v - lo) / (hi - lo)


def parse_claims(text: str) -> tuple[float | None, int, int]:
    """Retorna (fração de afirmações suportadas, n_afirmações, n_contraditas)."""
    obj = extract_json(text)
    if not obj or not isinstance(obj.get("claims"), list):
        return None, 0, 0
    verdicts = [str(c.get("veredito", "")).lower() for c in obj["claims"] if isinstance(c, dict)]
    if not verdicts:
        return None, 0, 0
    sup = sum(v.startswith("suportada") for v in verdicts)
    contra = sum(v.startswith("contradita") for v in verdicts)
    return sup / len(verdicts), len(verdicts), contra
