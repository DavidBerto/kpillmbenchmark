"""Custo medido no benchmark e custo projetado para produção."""
from __future__ import annotations

import numpy as np
import pandas as pd

from kpibench.config import BenchmarkConfig
from kpibench.llm.pricing import projected_cost_per_call


def cost_latency_table(cfg: BenchmarkConfig, preds: pd.DataFrame) -> pd.DataFrame:
    pj = cfg.projection
    rows = []
    for (kpi, model), g in preds.groupby(["kpi", "model"]):
        spec = cfg.model(model)
        ok = g[g["latency_s"].notna()]
        in_tok = float((ok["input_tokens"] + ok["cached_tokens"] + ok["cache_write_tokens"]).mean()) if len(ok) else np.nan
        out_tok = float(ok["output_tokens"].mean()) if len(ok) else np.nan
        prefix = float(g["static_prefix_tokens"].mean())
        proj_std = projected_cost_per_call(spec, in_tok, out_tok, prefix, pj.cache_hit_ratio, batch=False)
        proj_batch = projected_cost_per_call(spec, in_tok, out_tok, prefix, pj.cache_hit_ratio, batch=True)
        proj = proj_batch if pj.use_batch else proj_std
        rows.append({
            "kpi": kpi, "model": model, "tier": spec.tier, "n_calls": len(g),
            "error_rate": float((~g["valid"]).mean()),
            "mean_input_tokens": in_tok, "mean_output_tokens": out_tok,
            "mean_reasoning_tokens": float(ok["reasoning_tokens"].mean()) if len(ok) else np.nan,
            "static_prefix_tokens": prefix,
            "prefix_cacheable": prefix >= spec.min_cacheable_tokens,
            "measured_usd_per_1k": float(g["cost_usd"].mean() * 1000),
            "proj_usd_per_1k_standard": proj_std * 1000,
            "proj_usd_per_1k_batch": proj_batch * 1000,
            "proj_usd_per_1k": proj * 1000,
            "proj_usd_month": proj * pj.monthly_calls,
            "latency_p50_s": float(ok["latency_s"].median()) if len(ok) else np.nan,
            "latency_p95_s": float(ok["latency_s"].quantile(0.95)) if len(ok) else np.nan,
            "simulated": bool(g["simulated"].any()),
        })
    return pd.DataFrame(rows)


def evaluation_overhead(judge: pd.DataFrame | None, claims: pd.DataFrame | None) -> dict:
    """Custo do próprio benchmark com juízes (não entra no custo de produção)."""
    return {
        "judge_usd": float(judge["judge_cost_usd"].sum()) if judge is not None and not judge.empty else 0.0,
        "claims_usd": float(claims["cost_usd"].sum()) if claims is not None and not claims.empty else 0.0,
    }
