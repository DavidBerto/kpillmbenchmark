"""Construção dos clientes a partir da configuração."""
from __future__ import annotations

from kpibench.config import BenchmarkConfig, ModelSpec
from kpibench.data.loader import Dataset
from kpibench.llm.base import LLMClient


def build_client(spec: ModelSpec, cfg: BenchmarkConfig, dataset: Dataset | None = None) -> LLMClient:
    if cfg.run.mock or spec.provider == "mock":
        from kpibench.llm.providers.mock_provider import MockLLMClient

        sim = cfg.simulation or {}
        profile = sim.get(spec.id, {"skill": 0.8, "out_tokens": 200, "latency_s": 2.0})
        pool: dict[str, list[str]] = {}
        if dataset is not None:
            for k in cfg.selected_kpis:
                if k.type == "text" and k.gt_column in dataset.df.columns:
                    pool[k.name] = dataset.df[k.gt_column].dropna().astype(str).tolist()
        return MockLLMClient(spec, profile, sim.get("kpi_difficulty", {}), pool, seed=cfg.run.seed)

    if spec.provider == "nexus":
        from kpibench.llm.providers.nexus_provider import NexusChatClient

        return NexusChatClient(spec)

    if spec.provider in ("azure_openai", "openai", "openai_compatible"):
        from kpibench.llm.providers.openai_provider import OpenAIChatClient

        return OpenAIChatClient(spec)

    if spec.provider == "anthropic":
        from kpibench.llm.providers.anthropic_provider import AnthropicClient

        return AnthropicClient(spec)

    raise ValueError(f"provider desconhecido: {spec.provider}")
