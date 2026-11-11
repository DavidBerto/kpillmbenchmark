"""Backends de extração de KPIs para PRODUÇÃO, com uma interface única (`Extractor.extract`).

Cada backend recebe uma ligação ({"call_id", "transcricao", "resumo", ...}) e devolve uma `Extraction`:
valores por KPI (validados no domínio do KPI, com confiança quando o backend informa), as requisições
feitas (tokens, custo, latência) e a latência da ligação. Os backends são configurados em
`config/extractors.yaml`:

  llm_per_kpi      sistema atual: 1 chamada de LLM por KPI (prompts/kpis/<kpi>.md, sobre o resumo)
  llm_single_call  1 chamada de LLM por ligação, todos os KPIs num JSON pedido pelo prompt
  llm_json_schema  LLM com structured outputs: 1 chamada por ligação com a geração restrita a um JSON
                   Schema estrito gerado dos KPIs (decodificação restrita no servidor do LLM)
  cdm              ScaleDown CDM (Constrained Decoding Models): Decisions API para KPIs categóricos
                   (choice) e ordinais (score) numa requisição; Summarize para KPIs de texto livre
  jev              TypeSafe Jev: perguntas tipadas (choice/score) numa requisição por ligação; não gera
                   texto -> KPIs de texto ficam fora
  hybrid           primário de decisão (Jev ou CDM) nos KPIs estruturados + secundário (LLM) para KPIs
                   de texto e para os KPIs com confiança do primário abaixo do limiar

Custo de cada requisição = usage × preço (`llm.pricing.cost_usd`), ou o custo informado pela própria
API quando existe (usage.cost da ScaleDown); preços dos LLMs vêm de config/models.yaml e os de
CDM/Jev de config/extractors.yaml.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from kpibench import single_call as sc
from kpibench.config import BenchmarkConfig, KPISpec, ModelSpec, Pricing
from kpibench.data.loader import Dataset, normalize_label
from kpibench.extraction.parser import ParsedOutput, extract_json, parse_all_kpis_output, parse_kpi_output
from kpibench.extraction.runner import CallResult, LLMExecutor, _run_parallel
from kpibench.llm.pricing import cost_usd
from kpibench.llm.tokens import estimate_tokens
from kpibench.prompts.renderer import PromptRenderer, RenderedPrompt

STRUCTURED_TYPES = ("categorical", "ordinal")
REASONING_OVERHEAD = 300


# =============================================================================== resultado
@dataclass
class KPIValue:
    value: Any = None
    valid: bool = False
    error: str | None = None
    confidence: float | None = None     # Jev: confiança da resposta (0-1); LLM/CDM: None
    source: str = ""                    # backend (ou parte do híbrido) que produziu o valor
    justificativa: str | None = None


@dataclass
class RequestRecord:
    backend: str
    part: str                           # sub-backend que fez a requisição (no híbrido: jev ou cdm)
    model_id: str
    call_id: str
    n_kpis: int
    input_tokens: int = 0
    cached_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float | None = None
    prefix_tokens: int = 0              # parte estática da entrada (instruções), candidata a cache
    finish_reason: str | None = None
    error: str | None = None
    simulated: bool = False
    from_cache: bool = False


@dataclass
class Extraction:
    call_id: str
    backend: str
    values: dict[str, KPIValue]
    requests: list[RequestRecord]
    latency_s: float | None             # latência da ligação em produção (sequencial ou paralela)
    notes: dict = field(default_factory=dict)

    @property
    def cost_usd(self) -> float:
        return float(sum(r.cost_usd for r in self.requests))

    def tokens(self, col: str) -> int:
        return int(sum(getattr(r, col) for r in self.requests))

    def to_json(self) -> dict:
        """Saída de produção: um JSON por ligação com todos os KPIs."""
        kpis = {}
        for k, v in self.values.items():
            item = {"valor": v.value, "valido": v.valid}
            if v.confidence is not None:
                item["confianca"] = round(float(v.confidence), 4)
            if v.justificativa:
                item["justificativa"] = v.justificativa
            if v.error:
                item["erro"] = v.error
            item["fonte"] = v.source
            kpis[k] = item
        return {"call_id": self.call_id, "backend": self.backend, "kpis": kpis,
                "uso": {"requisicoes": len(self.requests), "tokens_entrada": self.tokens("input_tokens")
                        + self.tokens("cached_tokens") + self.tokens("cache_write_tokens"),
                        "tokens_saida": self.tokens("output_tokens"), "custo_usd": round(self.cost_usd, 8)},
                "latencia_s": None if self.latency_s is None else round(float(self.latency_s), 4),
                **({"notas": self.notes} if self.notes else {})}


# =============================================================================== contexto
class ExtractorContext:
    """Configuração privada (os backends registram os próprios ModelSpecs), renderer e executor com cache."""

    def __init__(self, cfg: BenchmarkConfig, dataset: Dataset | None = None, conf: dict | None = None):
        self.cfg = cfg.model_copy(deep=True)
        self.renderer = PromptRenderer(cfg.root / "prompts")
        self.dataset = dataset
        self.executor = LLMExecutor(self.cfg, dataset)
        self.conf = conf if conf is not None else load_extractor_config(cfg)
        self.built: dict[str, Extractor] = {}

    def register(self, spec: ModelSpec) -> None:
        self.cfg.model_registry[spec.id] = spec


def load_extractor_config(cfg: BenchmarkConfig, path: str | Path | None = None) -> dict:
    p = Path(path) if path else cfg.root / "config" / "extractors.yaml"
    with open(p, encoding="utf-8") as fh:
        return (yaml.safe_load(fh) or {}).get("extractors", {})


def _record(backend: str, part: str, spec: ModelSpec, res: CallResult, call_id: str, n_kpis: int,
            prefix_tokens: int) -> RequestRecord:
    r = RequestRecord(backend=backend, part=part, model_id=spec.id, call_id=call_id, n_kpis=n_kpis,
                      prefix_tokens=prefix_tokens, error=res.error, from_cache=res.cached)
    if res.response is not None:
        u = res.response.usage
        r.input_tokens, r.cached_tokens, r.cache_write_tokens = u.input_tokens, u.cached_tokens, u.cache_write_tokens
        r.output_tokens, r.reasoning_tokens = u.output_tokens, u.reasoning_tokens
        reported = getattr(res.response, "reported_cost_usd", None)
        r.cost_usd = reported if reported is not None else cost_usd(spec, u)
        r.latency_s = res.response.latency_s
        r.finish_reason = res.response.finish_reason
        r.simulated = res.response.simulated
    return r


def _text(call: dict, key: str) -> str:
    v = call.get(key)
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)


# =============================================================================== interface
class Extractor(ABC):
    kind = ""

    def __init__(self, name: str, ctx: ExtractorContext, kpis: list[KPISpec]):
        self.name, self.ctx, self.kpis = name, ctx, list(kpis)

    def supports(self, kpi: KPISpec) -> bool:
        return True

    @property
    def supported_kpis(self) -> list[KPISpec]:
        return [k for k in self.kpis if self.supports(k)]

    @abstractmethod
    def extract(self, call: dict, kpis: list[KPISpec] | None = None) -> Extraction: ...

    @abstractmethod
    def estimate(self, call: dict, kpis: list[KPISpec] | None = None) -> list[tuple[str, float, float]]:
        """Requisições previstas: [(model_id, tokens_entrada, tokens_saida)], sem chamar a API."""

    def failed(self, call: dict, error: str, kpis: list[KPISpec] | None = None) -> Extraction:
        kp = [k for k in (kpis or self.kpis) if self.supports(k)]
        return Extraction(str(call["call_id"]), self.name, {k.name: KPIValue(None, False, error, source=self.name)
                                                            for k in kp}, [], None)


# =============================================================================== LLM: 1 chamada por KPI
class LLMPerKPIExtractor(Extractor):
    kind = "llm_per_kpi"

    def __init__(self, name, ctx, kpis, model: str, input: str = "resumo", parallel_kpis: bool = True):
        super().__init__(name, ctx, kpis)
        self.spec = ctx.cfg.model(model)
        self.input, self.parallel = input, parallel_kpis

    def _prompt(self, k: KPISpec, call: dict) -> RenderedPrompt:
        t = _text(call, "transcricao")
        return self.ctx.renderer.render(f"kpis/{k.prompt}", resumo=_text(call, self.input), transcricao=t,
                                        has_transcript=bool(t), labels=k.labels or {},
                                        scale=k.scale.model_dump() if k.scale else None)

    def extract(self, call, kpis=None):
        cid = str(call["call_id"])
        values, reqs = {}, []
        for k in kpis or self.kpis:
            prompt = self._prompt(k, call)
            res = self.ctx.executor.call(self.spec.id, prompt, {"task": "kpi", "kpi": k.name, "kpi_spec": k,
                                                                 "call_id": cid, "record": call})
            p = parse_kpi_output(res.response.text, k) if res.response else ParsedOutput(None, False, res.error)
            values[k.name] = KPIValue(p.value, p.valid, res.error or p.error, None, self.name,
                                      p.extras.get("justificativa"))
            reqs.append(_record(self.name, self.name, self.spec, res, cid, 1, prompt.static_prefix_tokens))
        lats = [r.latency_s for r in reqs if r.latency_s is not None]
        latency = (max(lats) if self.parallel else sum(lats)) if lats else None
        return Extraction(cid, self.name, values, reqs, latency)

    def estimate(self, call, kpis=None):
        reasoning = REASONING_OVERHEAD if "reasoning_effort" in self.spec.params else 0
        out = []
        for k in kpis or self.kpis:
            p = self._prompt(k, call)
            o = sc.OUT_SUMMARY_QUALITY if k.name == "qualidade_resumo" else sc.OUT_PER_KPI[k.type]
            out.append((self.spec.id, estimate_tokens(p.system) + estimate_tokens(p.user), o + reasoning))
        return out


# =============================================================================== LLM: 1 chamada por ligação
class LLMSingleCallExtractor(Extractor):
    kind = "llm_single_call"

    def __init__(self, name, ctx, kpis, model: str | None = None, max_tokens: int | None = 4000,
                 justificativas: bool = True, max_palavras: int = 25, spec: ModelSpec | None = None):
        super().__init__(name, ctx, kpis)
        if spec is None:  # cópia do modelo de models.yaml com max_tokens maior (a saída traz todos os KPIs)
            base = ctx.cfg.model(model)
            params = {**base.params, **({"max_tokens": max_tokens} if max_tokens else {})}
            spec = base.model_copy(update={"id": f"{base.id}@{name}", "params": params})
        ctx.register(spec)
        self.spec, self.justificativas, self.max_palavras = spec, justificativas, max_palavras

    def request_params(self, kpis: list[KPISpec]) -> dict:
        return {}

    def _prompt(self, call, kp):
        return sc.render_single_call(self.ctx.renderer, kp, _text(call, "transcricao"), _text(call, "resumo"),
                                     self.max_palavras, self.justificativas)

    def extract(self, call, kpis=None):
        cid = str(call["call_id"])
        kp = list(kpis or self.kpis)
        prompt = self._prompt(call, kp)
        res = self.ctx.executor.call(self.spec.id, prompt, {"task": "kpi_all", "kpi_specs": kp, "call_id": cid,
                                                             "record": call}, params=self.request_params(kp))
        if res.response is not None:
            _, parsed = parse_all_kpis_output(res.response.text, kp)
        else:
            parsed = {k.name: ParsedOutput(None, False, res.error) for k in kp}
        values = {k: KPIValue(p.value, p.valid, p.error, None, self.name, p.extras.get("justificativa"))
                  for k, p in parsed.items()}
        rec = _record(self.name, self.name, self.spec, res, cid, len(kp), prompt.static_prefix_tokens)
        notes = {"truncado": True} if rec.finish_reason == "length" else {}
        return Extraction(cid, self.name, values, [rec], rec.latency_s, notes)

    def estimate(self, call, kpis=None):
        kp = list(kpis or self.kpis)
        p = self._prompt(call, kp)
        reasoning = REASONING_OVERHEAD if "reasoning_effort" in self.spec.params else 0
        out = 0.75 * sum(sc.OUT_SUMMARY_QUALITY if k.name == "qualidade_resumo" else sc.OUT_PER_KPI[k.type] for k in kp)
        return [(self.spec.id, estimate_tokens(p.system) + estimate_tokens(p.user), out + reasoning)]


# =============================================================================== LLM com structured outputs
def kpi_json_schema(kpis: list[KPISpec], justificativas: bool = True) -> dict:
    """JSON Schema ESTRITO da saída consolidada: todo objeto com additionalProperties=false e todas as
    propriedades obrigatórias (exigência do modo strict); domínio do KPI por enum."""
    props = {}
    for k in kpis:
        item, req = {}, []
        if k.name == "qualidade_resumo":
            item["alucinacoes"] = {"type": "array", "items": {"type": "string"}}
            req.append("alucinacoes")
        if justificativas and k.type != "text":
            item["justificativa"] = {"type": "string"}
            req.append("justificativa")
        if k.type == "categorical":
            item["valor"] = {"type": "string", "enum": list(k.labels or {})}
        elif k.type == "ordinal":
            item["valor"] = {"type": "integer", "enum": list(range(k.scale.min, k.scale.max + 1))}
        else:
            item["valor"] = {"type": "string"}
        req.append("valor")
        props[k.name] = {"type": "object", "additionalProperties": False, "required": req, "properties": item}
    return {"type": "object", "additionalProperties": False, "required": ["kpis"],
            "properties": {"kpis": {"type": "object", "additionalProperties": False,
                                    "required": [k.name for k in kpis], "properties": props}}}


class StructuredOutputLLMExtractor(LLMSingleCallExtractor):
    """LLM com structured outputs: mesmo prompt consolidado, geração restrita ao JSON Schema dos KPIs pedidos
    (decodificação restrita feita pelo servidor do LLM: response_format json_schema strict ou guided_json)."""
    kind = "llm_json_schema"

    def __init__(self, name, ctx, kpis, conf: dict):
        provider = conf.get("provider", "nexus")
        if provider == "nexus":   # modelo de config/models.yaml via Nexus, com o schema estrito por requisição
            base = ctx.cfg.model(conf["model"])
            spec = base.model_copy(update={"id": f"{base.id}@{name}", "json_mode": False,
                                           "params": {**base.params, "max_tokens": conf.get("max_tokens", 4000)}})
        else:                     # endpoint OpenAI-compatível próprio (OpenAI, Azure, vLLM, SGLang...)
            model = os.environ.get(conf.get("model_env") or "", "") or conf["model"]
            spec = ModelSpec(id=name, provider=provider, model=model, tier="llm_json_schema",
                             pricing=Pricing(**conf["pricing"]), batch_discount=conf.get("batch_discount", 0.0),
                             min_cacheable_tokens=conf.get("min_cacheable_tokens", 1024), json_mode=False,
                             params={**conf.get("params", {}), "max_tokens": conf.get("max_tokens", 4000)},
                             base_url=os.environ.get(conf.get("base_url_env") or "", "") or conf.get("base_url"),
                             api_key_env=conf.get("api_key_env"), pricing_source=conf.get("pricing_source", ""),
                             pricing_is_estimate=conf.get("pricing_is_estimate", False))
        super().__init__(name, ctx, kpis, justificativas=conf.get("justificativas", True),
                         max_palavras=conf.get("max_palavras", 25), spec=spec)
        self.schema_mode = conf.get("schema_mode", "json_schema")

    def request_params(self, kpis):
        schema = kpi_json_schema(kpis, self.justificativas)
        if self.schema_mode == "guided_json":
            return {"extra_body": {"guided_json": schema}}
        return {"response_format": {"type": "json_schema",
                                    "json_schema": {"name": "kpis_ligacao", "strict": True, "schema": schema}}}


# =============================================================================== modelos de decisão (Jev, ScaleDown)
def _decision_spec(name: str, provider: str, conf: dict, defaults: dict) -> ModelSpec:
    return ModelSpec(id=name, provider=provider, model=conf.get("model", defaults["model"]), tier="decision",
                     pricing=Pricing(**conf["pricing"]), base_url=conf.get("base_url", defaults["base_url"]),
                     api_key_env=conf.get("api_key_env", defaults["api_key_env"]), supports_temperature=False,
                     min_cacheable_tokens=10 ** 9,
                     params={"path": conf.get("path", defaults["path"]), "timeout_s": conf.get("timeout_s", 30),
                             "long_context": conf.get("long_context", defaults.get("long_context", False))},
                     pricing_source=conf.get("pricing_source", ""), pricing_is_estimate=conf.get("pricing_is_estimate", False))


class DecisionModelExtractor(Extractor):
    """Base dos modelos de decisão tipada (TypeSafe Jev, ScaleDown CDM Decisions): uma requisição por ligação
    com uma pergunta por KPI — choice (categóricos, critérios = rótulos) e score (ordinais, níveis da escala);
    a rubrica do KPI vai em `instructions`. Devolvem confiança por resposta e não geram texto."""
    task = "decisions"
    source_hint = "a transcrição completa da ligação"

    def __init__(self, name, ctx, kpis, spec: ModelSpec, score_levels: dict | None = None):
        super().__init__(name, ctx, kpis)
        self.spec = spec
        ctx.register(spec)
        self.levels = score_levels or {}
        self._questions = {k.name: self.question(k) for k in kpis if k.type in STRUCTURED_TYPES}

    def supports(self, kpi):
        return kpi.type in STRUCTURED_TYPES

    def question(self, k: KPISpec) -> dict:
        src = (f"Fonte: {self.source_hint}"
               + ("; avalie o RESUMO contra a transcrição" if k.name == "qualidade_resumo" else "")
               + ". Onde o texto abaixo disser 'resumo', leia 'transcrição'"
               + (", exceto ao avaliar o próprio resumo." if k.name == "qualidade_resumo" else "."))
        instructions = src + "\n\n" + sc.kpi_rubric(self.ctx.renderer, k)
        if k.type == "categorical":
            return {"type": "choice", "instructions": instructions, "criteria": dict(k.labels or {})}
        levels = self.levels.get(k.name) or [f"nota {v} (de {k.scale.min} a {k.scale.max})"
                                             for v in range(k.scale.min, k.scale.max + 1)]
        if len(levels) != k.scale.max - k.scale.min + 1:
            raise ValueError(f"{self.name}.score_levels.{k.name}: {len(levels)} níveis para a escala "
                             f"{k.scale.min}-{k.scale.max}")
        return {"type": "score", "instructions": instructions, "criteria": list(levels)}

    def state(self, call: dict, kpis: list[KPISpec]):
        state = {"transcricao": _text(call, "transcricao")}
        if any(k.name == "qualidade_resumo" for k in kpis):
            state["resumo"] = _text(call, "resumo")
        return state

    def payload(self, call: dict, kpis: list[KPISpec]) -> dict:
        return {"state": self.state(call, kpis), "questions": {k.name: self._questions[k.name] for k in kpis}}

    @staticmethod
    def parse_answers(text: str, kpis: list[KPISpec], source: str = "") -> dict[str, KPIValue]:
        obj = extract_json(text) if text else None
        answers = (obj or {}).get("answers") if isinstance((obj or {}).get("answers"), dict) else None
        out = {}
        for k in kpis:
            a = (answers or {}).get(k.name)
            if not isinstance(a, dict):
                out[k.name] = KPIValue(None, False, "json_invalido" if answers is None else f"campo_ausente:{k.name}",
                                       source=source)
                continue
            conf = a.get("confidence")
            conf = float(conf) if isinstance(conf, (int, float)) else None
            if k.type == "categorical":
                v = normalize_label(a.get("choice"))
                ok = v in (k.labels or {})
                out[k.name] = KPIValue(v if ok else None, ok, None if ok else f"rotulo_invalido:{a.get('choice')}",
                                       conf, source)
            else:
                try:
                    s = float(a.get("score"))
                except (TypeError, ValueError):
                    out[k.name] = KPIValue(None, False, f"score_invalido:{a.get('score')}", conf, source)
                    continue
                v = int(np.clip(round(s), 0, k.scale.max - k.scale.min)) + k.scale.min
                out[k.name] = KPIValue(v, True, None, conf, source)
        return out

    def _prompt(self, call, kp) -> tuple[RenderedPrompt, int]:
        payload = self.payload(call, kp)
        questions = json.dumps(payload["questions"], ensure_ascii=False, sort_keys=True)
        h = hashlib.sha256(questions.encode()).hexdigest()[:12]
        prompt = RenderedPrompt(system="", user=json.dumps(payload, ensure_ascii=False, sort_keys=True),
                                template=f"{self.kind}/decisions", template_hash=h)
        return prompt, estimate_tokens(questions)

    def _decide(self, call, kp) -> tuple[dict[str, KPIValue], RequestRecord | None]:
        if not kp:
            return {}, None
        cid = str(call["call_id"])
        prompt, prefix = self._prompt(call, kp)
        res = self.ctx.executor.call(self.spec.id, prompt, {"task": self.task, "kpi_specs": kp, "call_id": cid,
                                                             "record": call})
        values = (self.parse_answers(res.response.text, kp, self.name) if res.response
                  else {k.name: KPIValue(None, False, res.error, source=self.name) for k in kp})
        return values, _record(self.name, self.name, self.spec, res, cid, len(kp), prefix)

    def extract(self, call, kpis=None):
        kp = [k for k in (kpis or self.kpis) if k.type in STRUCTURED_TYPES]
        values, rec = self._decide(call, kp)
        return Extraction(str(call["call_id"]), self.name, values, [rec] if rec else [], rec.latency_s if rec else None)

    def estimate(self, call, kpis=None):
        kp = [k for k in (kpis or self.kpis) if k.type in STRUCTURED_TYPES]
        if not kp:
            return []
        prompt, _ = self._prompt(call, kp)
        return [(self.spec.id, estimate_tokens(prompt.user), 4 * len(kp))]


class JevExtractor(DecisionModelExtractor):
    """TypeSafe Jev (System One). Não gera texto: cobre só KPIs categóricos e ordinais."""
    kind = "jev"
    task = "jev"
    source_hint = "a transcrição completa da ligação (state.transcricao)"

    def __init__(self, name, ctx, kpis, conf: dict):
        spec = _decision_spec(name, "typesafe", conf, {"model": "jev-latest", "base_url": "https://api.typesafe.ai",
                                                       "path": "/v1/systemone", "api_key_env": "TYPESAFE_API_KEY",
                                                       "long_context": "auto"})
        super().__init__(name, ctx, kpis, spec, conf.get("score_levels"))


class ScaleDownCDMExtractor(DecisionModelExtractor):
    """ScaleDown Constrained Decoding Models: Decisions API (choice/score) para os KPIs categóricos e ordinais
    numa requisição, e Summarize (abstrativo, guiado por instrução) para cada KPI de texto livre. As
    requisições são independentes e saem em paralelo em produção (latência = a mais lenta)."""
    kind = "cdm"
    task = "decisions"
    source_hint = "a transcrição completa da ligação (seção TRANSCRIÇÃO do texto)"

    def __init__(self, name, ctx, kpis, conf: dict):
        spec = _decision_spec(name, "scaledown", conf, {"model": "classify-1", "base_url": "https://api.scaledown.xyz",
                                                        "path": "/v1/scaledown", "api_key_env": "SCALEDOWN_API_KEY"})
        super().__init__(name, ctx, kpis, spec, conf.get("score_levels"))
        sm = conf.get("summarize") or {}
        self.text_mode = sm.get("enabled", True)
        self.parallel = conf.get("parallel_requests", True)
        self.summ_spec = ModelSpec(
            id=f"{name}@summarize", provider="openai_compatible", model=sm.get("model", "summarize-1"),
            tier="cdm", pricing=Pricing(**sm.get("pricing", conf["pricing"])),
            base_url=sm.get("base_url", spec.base_url.rstrip("/") + "/v1"), api_key_env=spec.api_key_env,
            supports_temperature=False, json_mode=False, min_cacheable_tokens=10 ** 9,
            params={"max_tokens": sm.get("max_tokens", 150)}, pricing_source=spec.pricing_source,
            pricing_is_estimate=spec.pricing_is_estimate)
        ctx.register(self.summ_spec)
        self.max_palavras = sm.get("max_palavras", 40)

    def supports(self, kpi):
        return kpi.type in STRUCTURED_TYPES or (kpi.type == "text" and self.text_mode)

    def state(self, call, kpis):
        text = "TRANSCRIÇÃO:\n" + _text(call, "transcricao")
        if any(k.name == "qualidade_resumo" for k in kpis):
            text += "\n\nRESUMO:\n" + _text(call, "resumo")
        return {"text": text}

    def summarize_prompt(self, call, k: KPISpec) -> RenderedPrompt:
        rubric = sc.kpi_rubric(self.ctx.renderer, k)
        system = (f"Responda em português, em no máximo {self.max_palavras} palavras, só com o que está na "
                  f"transcrição. Onde o texto abaixo disser 'resumo', leia 'transcrição'.\n\n{rubric}")
        return RenderedPrompt(system=system, user=_text(call, "transcricao"), template="cdm/summarize",
                              template_hash=hashlib.sha256(system.encode()).hexdigest()[:12])

    @staticmethod
    def parse_summary(text: str, kpi: KPISpec, source: str) -> KPIValue:
        obj = extract_json(text) if text else None
        s = (obj or {}).get("summary") if obj else None
        s = str(s).strip() if s else ""
        if not s:
            return KPIValue(None, False, "json_invalido" if obj is None else "texto_vazio", source=source)
        return KPIValue(s, True, None, None, source)

    def extract(self, call, kpis=None):
        cid = str(call["call_id"])
        kp = [k for k in (kpis or self.kpis) if self.supports(k)]
        values, rec = self._decide(call, [k for k in kp if k.type in STRUCTURED_TYPES])
        reqs = [rec] if rec else []
        for k in (k for k in kp if k.type == "text"):
            prompt = self.summarize_prompt(call, k)
            res = self.ctx.executor.call(self.summ_spec.id, prompt, {"task": "summarize", "kpi_spec": k,
                                                                      "call_id": cid, "record": call})
            values[k.name] = (self.parse_summary(res.response.text, k, self.name) if res.response
                              else KPIValue(None, False, res.error, source=self.name))
            r = _record(self.name, self.name, self.summ_spec, res, cid, 1, prompt.static_prefix_tokens)
            # o Summarize informa finish_reason dentro do JSON do conteúdo
            if res.response is not None and '"finish_reason": "length"' in res.response.text.replace('":"', '": "'):
                r.finish_reason = "length"
            reqs.append(r)
        lats = [r.latency_s for r in reqs if r.latency_s is not None]
        latency = (max(lats) if self.parallel else sum(lats)) if lats else None
        notes = {"truncado": True} if any(r.finish_reason == "length" for r in reqs) else {}
        return Extraction(cid, self.name, values, reqs, latency, notes)

    def estimate(self, call, kpis=None):
        kp = [k for k in (kpis or self.kpis) if self.supports(k)]
        out = super().estimate(call, kp)
        for k in (k for k in kp if k.type == "text"):
            p = self.summarize_prompt(call, k)
            out.append((self.summ_spec.id, estimate_tokens(p.system) + estimate_tokens(p.user), 60))
        return out


# =============================================================================== híbrido
class HybridExtractor(Extractor):
    """Primário de decisão (Jev ou CDM) nos KPIs de `primary_types` + secundário (LLM) para os demais KPIs
    (texto livre) e para os KPIs em que o primário respondeu com confiança abaixo do limiar."""
    kind = "hybrid"

    def __init__(self, name, ctx, kpis, primary: Extractor, secondary: Extractor, confidence_threshold: float = 0.3,
                 primary_types: tuple[str, ...] = STRUCTURED_TYPES):
        super().__init__(name, ctx, kpis)
        self.primary, self.secondary, self.threshold = primary, secondary, confidence_threshold
        self.primary_types = tuple(primary_types)
        self.kind = f"hybrid_{primary.kind}"

    def _on_primary(self, k: KPISpec) -> bool:
        return self.primary.supports(k) and k.type in self.primary_types

    def extract(self, call, kpis=None):
        cid = str(call["call_id"])
        kp = list(kpis or self.kpis)
        prim = [k for k in kp if self._on_primary(k)]
        r1 = self.primary.extract(call, prim) if prim else None
        low = [k for k in prim if not r1.values[k.name].valid
               or (r1.values[k.name].confidence is not None and r1.values[k.name].confidence < self.threshold)]
        need = [k for k in kp if not self._on_primary(k)] + low
        r2 = self.secondary.extract(call, need) if need else None
        low_names = {k.name for k in low}
        values = {}
        for k in kp:
            if r2 is not None and k.name in r2.values and (k.name in low_names or not self._on_primary(k)):
                v = r2.values[k.name]
                v.source = self.secondary.name + (" (fallback)" if k.name in low_names else "")
                if k.name in low_names:
                    v.confidence = r1.values[k.name].confidence   # confiança do primário que disparou o fallback
            else:
                v = r1.values[k.name]
            values[k.name] = v
        reqs = []
        for r in (r1, r2):
            for q in (r.requests if r else []):
                q.backend = self.name
                reqs.append(q)
        lats = [r.latency_s for r in (r1, r2) if r is not None and r.latency_s is not None]
        notes = {"fallback_kpis": sorted(low_names)} if low_names else {}
        return Extraction(cid, self.name, values, reqs, sum(lats) if lats else None, notes)

    def estimate(self, call, kpis=None):
        kp = list(kpis or self.kpis)
        prim = [k for k in kp if self._on_primary(k)]
        rest = [k for k in kp if not self._on_primary(k)]
        return (self.primary.estimate(call, prim) if prim else []) + (self.secondary.estimate(call, rest) if rest else [])


# =============================================================================== fábrica e execução
def build_extractor(name: str, ctx: ExtractorContext, kpis: list[KPISpec]) -> Extractor:
    if name in ctx.built:
        return ctx.built[name]
    if name not in ctx.conf:
        raise KeyError(f"backend '{name}' não está em config/extractors.yaml (disponíveis: {sorted(ctx.conf)})")
    c = ctx.conf[name]
    t = c["type"]
    if t == "llm_per_kpi":
        ex = LLMPerKPIExtractor(name, ctx, kpis, c["model"], c.get("input", "resumo"), c.get("parallel_kpis", True))
    elif t == "llm_single_call":
        ex = LLMSingleCallExtractor(name, ctx, kpis, c["model"], c.get("max_tokens", 4000),
                                    c.get("justificativas", True), c.get("max_palavras", 25))
    elif t == "llm_json_schema":
        ex = StructuredOutputLLMExtractor(name, ctx, kpis, c)
    elif t == "cdm":
        ex = ScaleDownCDMExtractor(name, ctx, kpis, c)
    elif t == "jev":
        ex = JevExtractor(name, ctx, kpis, c)
    elif t == "hybrid":
        ex = HybridExtractor(name, ctx, kpis, build_extractor(c["primary"], ctx, kpis),
                             build_extractor(c["secondary"], ctx, kpis), c.get("confidence_threshold", 0.3),
                             tuple(c.get("primary_types", STRUCTURED_TYPES)))
    else:
        raise ValueError(f"backend '{name}': tipo desconhecido '{t}'")
    ctx.built[name] = ex
    return ex


def credentials_status(ctx: ExtractorContext, names: list[str]) -> pd.DataFrame:
    """Variáveis de ambiente que cada backend precisa e se estão definidas (sem expor valores)."""
    rows = []

    def env_rows(name, c, via=""):
        t = c["type"]
        if t in ("llm_per_kpi", "llm_single_call") or (t == "llm_json_schema" and c.get("provider", "nexus") == "nexus"):
            needed = ["base_url", "bearer_token"]
        elif t == "llm_json_schema":
            needed = [c.get("api_key_env"), c.get("base_url_env")]
        elif t == "cdm":
            needed = [c.get("api_key_env", "SCALEDOWN_API_KEY")]
        elif t == "jev":
            needed = [c.get("api_key_env", "TYPESAFE_API_KEY")]
        else:
            for sub in (c["primary"], c["secondary"]):
                env_rows(name, ctx.conf[sub], via=sub)
            return
        for e in filter(None, needed):
            rows.append({"backend": name, "via": via or name, "variavel": e, "definida": bool(os.environ.get(e))})

    for n in names:
        env_rows(n, ctx.conf[n])
    return pd.DataFrame(rows).drop_duplicates()


def run_extractor(extractor: Extractor, calls: list[dict], workers: int = 8) -> list[Extraction]:
    jobs = []
    for call in calls:
        def job(call=call) -> Extraction:
            try:
                return extractor.extract(call)
            except Exception as e:  # noqa: BLE001 - falha da ligação vira KPIs inválidos, não derruba o lote
                return extractor.failed(call, f"{type(e).__name__}: {e}"[:300])
        jobs.append(job)
    print(f"Extração [{extractor.name}]: {len(jobs)} ligações", file=sys.stderr)
    return _run_parallel(jobs, workers, extractor.name)


def extraction_frames(exts: list[Extraction]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(por_ligacao, predicoes_longas, requisicoes)."""
    calls, preds, reqs = [], [], []
    for e in exts:
        n_valid = sum(v.valid for v in e.values.values())
        calls.append({"backend": e.backend, "call_id": e.call_id, "requisicoes": len(e.requests),
                      **{c: e.tokens(c) for c in ("input_tokens", "cached_tokens", "cache_write_tokens",
                                                  "output_tokens", "reasoning_tokens")},
                      "cost_usd": e.cost_usd, "latency_s": e.latency_s, "kpis_validos": n_valid,
                      "n_kpis": len(e.values), "fallback_kpis": len(e.notes.get("fallback_kpis", [])),
                      "truncado": bool(e.notes.get("truncado")),
                      "simulated": any(r.simulated for r in e.requests)})
        for k, v in e.values.items():
            preds.append({"model": e.backend, "kpi": k, "call_id": e.call_id, "value": v.value, "valid": v.valid,
                          "error": v.error, "confidence": v.confidence, "source": v.source})
        reqs += [asdict(r) for r in e.requests]
    return pd.DataFrame(calls), pd.DataFrame(preds), pd.DataFrame(reqs)


def write_jsonl(exts: list[Extraction], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for e in exts:
            fh.write(json.dumps(e.to_json(), ensure_ascii=False, default=str) + "\n")
    return path


def estimate_cost(extractor: Extractor, calls: list[dict]) -> dict:
    """Custo e tokens previstos para estas ligações (antes de executar)."""
    reqs = [r for c in calls for r in extractor.estimate(c)]
    usd = sum((i * extractor.ctx.cfg.model(m).pricing.input + o * extractor.ctx.cfg.model(m).pricing.output) / 1e6
              for m, i, o in reqs)
    n = max(len(calls), 1)
    return {"backend": extractor.name, "tipo": extractor.kind, "ligacoes": len(calls), "requisicoes": len(reqs),
            "entrada_tokens_por_ligacao": sum(r[1] for r in reqs) / n, "saida_tokens_por_ligacao": sum(r[2] for r in reqs) / n,
            "usd_estimado": usd, "kpis_cobertos": len(extractor.supported_kpis)
            if not isinstance(extractor, HybridExtractor) else len(extractor.kpis)}
