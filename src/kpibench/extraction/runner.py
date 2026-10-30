"""Execução concorrente: modelos x KPIs x registros (+ juízes), com cache e retries."""
from __future__ import annotations

import json
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable

import pandas as pd

from kpibench.config import BenchmarkConfig, KPISpec
from kpibench.data.loader import Dataset
from kpibench.extraction.parser import parse_claims, parse_judge_score, parse_kpi_output
from kpibench.llm.base import LLMClient, LLMRequest, LLMResponse, TransientLLMError
from kpibench.llm.cache import ResponseCache, cache_key
from kpibench.llm.factory import build_client
from kpibench.llm.pricing import cost_usd
from kpibench.prompts.renderer import PromptRenderer, RenderedPrompt


@dataclass
class CallResult:
    response: LLMResponse | None
    error: str | None
    cached: bool = False


class LLMExecutor:
    """Encapsula clientes, cache e política de retry."""

    def __init__(self, cfg: BenchmarkConfig, dataset: Dataset | None = None):
        self.cfg = cfg
        self.dataset = dataset
        self.cache = ResponseCache(cfg.resolve(cfg.run.cache_path) if cfg.run.cache_path else None)
        self._clients: dict[str, LLMClient] = {}
        self._lock = threading.Lock()

    def client(self, model_id: str) -> LLMClient:
        with self._lock:
            if model_id not in self._clients:
                self._clients[model_id] = build_client(self.cfg.model(model_id), self.cfg, self.dataset)
            return self._clients[model_id]

    def call(self, model_id: str, prompt: RenderedPrompt, metadata: dict) -> CallResult:
        cli = self.client(model_id)
        key = cache_key(model_id, cli.spec.model, cli.generation_params(), prompt.system, prompt.user)
        hit = self.cache.get(key)
        if hit is not None:
            return CallResult(hit, None, cached=True)
        req = LLMRequest(system=prompt.system, user=prompt.user, metadata=metadata)
        last_err = None
        for attempt in range(self.cfg.run.max_retries + 1):
            try:
                resp = cli.complete(req)
                self.cache.put(key, resp)
                return CallResult(resp, None)
            except TransientLLMError as e:
                last_err = f"transiente: {e}"
                time.sleep(min(60.0, (2 ** attempt) + random.random()))
            except Exception as e:  # noqa: BLE001 - erro definitivo vira linha inválida
                return CallResult(None, f"{type(e).__name__}: {e}"[:300])
        return CallResult(None, last_err)


def _run_parallel(jobs: list[Callable[[], dict]], workers: int, label: str) -> list[dict]:
    rows: list[dict] = []
    total = len(jobs)
    if total == 0:
        return rows
    step = max(1, total // 10)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(j) for j in jobs]
        for i, f in enumerate(as_completed(futures), 1):
            rows.append(f.result())
            if i % step == 0 or i == total:
                print(f"  [{label}] {i}/{total}", file=sys.stderr)
    return rows


def _usage_cols(model_spec, res: CallResult) -> dict:
    if res.response is None:
        return {"input_tokens": 0, "cached_tokens": 0, "cache_write_tokens": 0, "output_tokens": 0,
                "reasoning_tokens": 0, "latency_s": None, "cost_usd": 0.0, "simulated": False,
                "from_cache": False}
    u = res.response.usage
    return {"input_tokens": u.input_tokens, "cached_tokens": u.cached_tokens,
            "cache_write_tokens": u.cache_write_tokens, "output_tokens": u.output_tokens,
            "reasoning_tokens": u.reasoning_tokens, "latency_s": res.response.latency_s,
            "cost_usd": cost_usd(model_spec, u), "simulated": res.response.simulated,
            "from_cache": res.cached}


def kpi_prompt_vars(kpi: KPISpec, ds: Dataset, rec: dict) -> dict:
    transcript = rec.get(ds.transcript_col) if ds.transcript_col else None
    has_t = bool(transcript) and not (isinstance(transcript, float))
    return {
        "resumo": str(rec[ds.summary_col]),
        "transcricao": str(transcript) if has_t else "",
        "has_transcript": has_t,
        "labels": kpi.labels or {},
        "scale": kpi.scale.model_dump() if kpi.scale else {},
    }


# =========================================================================== KPIs
def run_kpi_extraction(cfg: BenchmarkConfig, ds: Dataset, renderer: PromptRenderer,
                       ex: LLMExecutor) -> pd.DataFrame:
    jobs = []
    records = {cid: ds.record(cid) for cid in ds.ids()}
    for kpi in cfg.selected_kpis:
        for model_id in cfg.models:
            spec = cfg.model(model_id)
            for cid, rec in records.items():
                def job(kpi=kpi, model_id=model_id, spec=spec, cid=cid, rec=rec) -> dict:
                    prompt = renderer.render(f"kpis/{kpi.prompt}", **kpi_prompt_vars(kpi, ds, rec))
                    res = ex.call(model_id, prompt, {"task": "kpi", "kpi": kpi.name, "kpi_spec": kpi,
                                                     "call_id": cid, "record": rec})
                    parsed = parse_kpi_output(res.response.text, kpi) if res.response else None
                    return {
                        "model": model_id, "kpi": kpi.name, "call_id": cid,
                        "value": parsed.value if parsed else None,
                        "valid": bool(parsed and parsed.valid),
                        "error": res.error or (parsed.error if parsed else None),
                        "raw_output": (res.response.text[:2000] if res.response else None),
                        "extras": json.dumps(parsed.extras, ensure_ascii=False) if parsed and parsed.extras else None,
                        "prompt_hash": prompt.template_hash,
                        "static_prefix_tokens": prompt.static_prefix_tokens,
                        **_usage_cols(spec, res),
                    }
                jobs.append(job)
    print(f"Extração: {len(jobs)} chamadas ({len(cfg.models)} modelos x "
          f"{len(cfg.selected_kpis)} KPIs x {len(records)} registros)", file=sys.stderr)
    return pd.DataFrame(_run_parallel(jobs, cfg.run.max_workers, "extração"))


# =========================================================================== juiz de texto
def run_reasoning_judge(cfg: BenchmarkConfig, ds: Dataset, preds: pd.DataFrame,
                        renderer: PromptRenderer, ex: LLMExecutor) -> pd.DataFrame:
    """Nota 0-1 do juiz para cada justificativa candidata (KPIs de texto)."""
    if not cfg.judge.enabled or not cfg.judge.model:
        return pd.DataFrame(columns=["model", "kpi", "call_id", "judge_score"])
    judge_id = cfg.judge.model
    spec = cfg.model(judge_id)
    text_kpis = {k.name: k for k in cfg.selected_kpis if k.type == "text" and k.judge_prompt}
    sub = preds[preds["kpi"].isin(text_kpis) & preds["valid"]]
    jobs = []
    for row in sub.itertuples(index=False):
        kpi = text_kpis[row.kpi]
        rec = ds.record(row.call_id)
        ref = rec.get(kpi.gt_column)
        if ref is None or (isinstance(ref, float)):
            continue

        def job(row=row, kpi=kpi, rec=rec, ref=ref) -> dict:
            prompt = renderer.render(f"judges/{kpi.judge_prompt}", judge_context=kpi.judge_context,
                                     resumo=str(rec[ds.summary_col]), referencia=str(ref),
                                     candidata=str(row.value))
            res = ex.call(judge_id, prompt, {"task": "judge_reasoning", "kpi": kpi.name,
                                             "call_id": row.call_id, "candidate_model": row.model,
                                             "referencia": str(ref), "candidata": str(row.value)})
            score = parse_judge_score(res.response.text) if res.response else None
            u = _usage_cols(spec, res)
            return {"model": row.model, "kpi": row.kpi, "call_id": row.call_id,
                    "judge_score": score, "judge_cost_usd": u["cost_usd"], "judge_error": res.error}
        jobs.append(job)
    print(f"Juiz ({judge_id}): {len(jobs)} chamadas", file=sys.stderr)
    return pd.DataFrame(_run_parallel(jobs, cfg.run.max_workers, "juiz"))


# =========================================================================== fidelidade por claims
def run_claim_faithfulness(cfg: BenchmarkConfig, ds: Dataset, renderer: PromptRenderer,
                           ex: LLMExecutor) -> pd.DataFrame:
    judge_id = cfg.summary_metrics.claim_judge_model
    if not judge_id or not ds.has_transcript or "claim_faithfulness" not in cfg.summary_metrics.modern:
        return pd.DataFrame()
    spec = cfg.model(judge_id)
    jobs = []
    for cid in ds.ids():
        rec = ds.record(cid)

        def job(cid=cid, rec=rec) -> dict:
            prompt = renderer.render("judges/claim_faithfulness.md",
                                     transcricao=str(rec[ds.transcript_col]), resumo=str(rec[ds.summary_col]))
            res = ex.call(judge_id, prompt, {"task": "claim_faithfulness", "call_id": cid,
                                             "record": rec, "summary": str(rec[ds.summary_col])})
            frac, n, contra = parse_claims(res.response.text) if res.response else (None, 0, 0)
            u = _usage_cols(spec, res)
            return {"call_id": cid, "claim_faithfulness": frac, "n_claims": n, "n_contradicted": contra,
                    "cost_usd": u["cost_usd"], "latency_s": u["latency_s"], "error": res.error}
        jobs.append(job)
    print(f"Fidelidade por afirmações ({judge_id}): {len(jobs)} chamadas", file=sys.stderr)
    return pd.DataFrame(_run_parallel(jobs, cfg.run.max_workers, "claims"))
