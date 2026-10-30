"""Provider SIMULADO para o demo offline e testes.

Gera saídas no MESMO formato JSON que os prompts pedem, a partir do rótulo validado
do registro + ruído controlado por um perfil de habilidade (config `simulation`).
As saídas passam pelo mesmo parser/normalizador das rodadas reais. Tokens de entrada
são estimados do prompt renderizado de verdade; nenhuma latência real é gasta.

NÃO use resultados do mock para escolher modelo.
"""
from __future__ import annotations

import hashlib
import json
import re

import numpy as np

from kpibench.config import KPISpec, ModelSpec
from kpibench.llm.base import LLMClient, LLMRequest, LLMResponse, Usage
from kpibench.llm.tokens import estimate_tokens

_SYNONYMS = {
    "cliente": "consumidor", "atendente": "operador", "demanda": "solicitação",
    "resolvida": "solucionada", "insatisfeito": "descontente", "satisfeito": "contente",
    "cobrança": "lançamento", "prazo": "previsão", "informou": "explicou", "abriu": "registrou",
}


def _rng(*parts) -> np.random.Generator:
    h = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def _paraphrase(text: str, rng: np.random.Generator) -> str:
    words = text.split()
    out = []
    for w in words:
        key = w.lower().strip(".,;")
        if key in _SYNONYMS and rng.random() < 0.6:
            w = w.lower().replace(key, _SYNONYMS[key])
        out.append(w)
    s = " ".join(out)
    if rng.random() < 0.3 and "," in s:  # às vezes omite a última oração
        s = s.rsplit(",", 1)[0] + "."
    return s


class MockLLMClient(LLMClient):
    def __init__(self, spec: ModelSpec, profile: dict, kpi_difficulty: dict[str, float],
                 text_pool: dict[str, list[str]], seed: int = 0):
        super().__init__(spec)
        self.profile = profile
        self.difficulty = kpi_difficulty
        self.pool = text_pool
        self.seed = seed

    # ------------------------------------------------------------------ API
    def complete(self, req: LLMRequest) -> LLMResponse:
        md = req.metadata
        rng = _rng(self.seed, self.spec.id, md.get("task"), md.get("kpi"), md.get("call_id"),
                   md.get("candidate_model", ""))
        task = md["task"]
        if task == "kpi":
            payload = self._kpi_output(md["kpi_spec"], md["record"], rng)
        elif task == "judge_reasoning":
            payload = self._judge_output(md["referencia"], md["candidata"], rng)
        elif task == "claim_faithfulness":
            payload = self._claims_output(md["record"], md.get("summary", ""), rng)
        else:
            raise ValueError(task)

        text = json.dumps(payload, ensure_ascii=False)
        if rng.random() < self.profile.get("invalid_rate", 0.0):
            text = rng.choice([text[: len(text) // 2],
                               "Claro! Segue a análise solicitada sobre o atendimento."])
        elif rng.random() < self.profile.get("fence_rate", 0.1):
            text = f"```json\n{text}\n```"

        base_out = self.profile.get("out_tokens", 200)
        out_mult = 1.8 if md.get("kpi") == "qualidade_resumo" or task == "claim_faithfulness" else 1.0
        out_tokens = int(base_out * out_mult * rng.lognormal(0, 0.25))
        usage = Usage(input_tokens=estimate_tokens(req.system) + estimate_tokens(req.user),
                      output_tokens=max(out_tokens, estimate_tokens(text)))
        latency = float(self.profile.get("latency_s", 2.0) * out_mult * rng.lognormal(0, 0.3))
        return LLMResponse(text=text, usage=usage, latency_s=latency, model=self.spec.model,
                           finish_reason="stop", simulated=True)

    # ------------------------------------------------------------------ helpers
    def _p_correct(self, kpi_name: str) -> float:
        return float(np.clip(self.profile.get("skill", 0.8) - self.difficulty.get(kpi_name, 0.0), 0.05, 0.99))

    def _kpi_output(self, kpi: KPISpec, record: dict, rng: np.random.Generator) -> dict:
        gt = record.get(kpi.gt_column)
        p = self._p_correct(kpi.name)
        correct = rng.random() < p
        just = "Avaliação baseada nos elementos descritos no resumo."

        if kpi.type == "ordinal":
            lo, hi = kpi.scale.min, kpi.scale.max
            gt_i = int(gt) if gt is not None and not _isnan(gt) else (lo + hi) // 2
            if correct:
                v = gt_i
            else:
                step = 1 if rng.random() < 0.75 else 2
                v = int(np.clip(gt_i + step * rng.choice([-1, 1]), lo, hi))
                if v == gt_i:
                    v = int(np.clip(gt_i - step if gt_i == hi else gt_i + step, lo, hi))
            if kpi.name == "qualidade_resumo":
                dims = {d: int(np.clip(v + rng.integers(-1, 2), lo, hi))
                        for d in ("fidelidade", "cobertura", "relevancia", "coerencia", "fluencia")}
                return {"analise": "Resumo comparado à transcrição.", "alucinacoes": [], **dims,
                        "nota_final": v}
            return {"justificativa": just, kpi.output_field: v}

        if kpi.type == "categorical":
            labels = list(kpi.labels)
            if correct and gt in labels:
                v = gt
            else:
                others = [l for l in labels if l != gt]
                v = str(rng.choice(others))
            return {"justificativa": just, kpi.output_field: v}

        # text
        if correct and isinstance(gt, str) and gt:
            v = _paraphrase(gt, rng)
        else:
            pool = [t for t in self.pool.get(kpi.name, []) if t != gt] or ["Resumo não traz elementos suficientes."]
            v = str(rng.choice(pool))
        return {kpi.output_field: v}

    def _judge_output(self, ref: str, cand: str, rng: np.random.Generator) -> dict:
        from kpibench.metrics.text import chrf

        sim = chrf(cand or "", ref or "")
        base = 1 + 4 * np.clip((sim - 0.15) / 0.5, 0, 1)
        noise = rng.normal(0, 0.6 * (1 - self.profile.get("skill", 0.9)) + 0.2)
        nota = int(np.clip(round(base + noise), 1, 5))
        return {"analise": "Comparação com a referência.", "fundamentacao": nota,
                "concordancia": nota, "completude": nota, "nota": nota}

    def _claims_output(self, record: dict, summary: str, rng: np.random.Generator) -> dict:
        summary = str(summary)
        sents = [s.strip() for s in re.split(r"(?<=[.;])\s+", summary) if s.strip()]
        q = record.get("gt_qualidade_resumo")
        q = 3 if q is None or _isnan(q) else int(q)
        target = {5: 0.98, 4: 0.92, 3: 0.82, 2: 0.6, 1: 0.45}[q]
        target = float(np.clip(target + rng.normal(0, 0.25 * (1 - self.profile.get("skill", 0.9)) + 0.05), 0, 1))
        claims = []
        for s in sents:
            r = rng.random()
            ver = "suportada" if r < target else ("contradita" if r < target + (1 - target) / 2 else "nao_suportada")
            claims.append({"afirmacao": s[:120], "veredito": ver})
        return {"claims": claims}


def _isnan(v) -> bool:
    import pandas as pd

    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False
