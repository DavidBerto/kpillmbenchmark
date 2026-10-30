"""Carregamento e validação das configurações (benchmark + modelos + KPIs)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- modelos
class Pricing(BaseModel):
    input: float = Field(ge=0, description="USD por 1M tokens de entrada")
    cached_input: float = Field(ge=0, description="USD por 1M tokens lidos do cache")
    output: float = Field(ge=0, description="USD por 1M tokens de saída (inclui raciocínio)")
    cache_write_multiplier: float = 1.25  # Anthropic: escrita de cache 5 min = 1.25x input


class ModelSpec(BaseModel):
    id: str
    provider: Literal["nexus", "azure_openai", "openai", "anthropic", "openai_compatible", "mock"]
    model: str
    tier: str = ""
    pricing: Pricing
    batch_discount: float = 0.0
    min_cacheable_tokens: int = 1024
    supports_temperature: bool = True
    json_mode: bool = False
    params: dict[str, Any] = Field(default_factory=dict)
    base_url: str | None = None
    api_key_env: str | None = None
    pricing_source: str = ""
    verified_on: str = ""


# --------------------------------------------------------------------------- KPIs
class Scale(BaseModel):
    min: int
    max: int


class KPISpec(BaseModel):
    name: str
    display: str
    type: Literal["ordinal", "categorical", "text"]
    gt_column: str
    prompt: str
    output_field: str
    ni_margin: float = 0.03
    scale: Scale | None = None
    labels: dict[str, str] | None = None
    needs_transcript: bool = False
    judge_prompt: str | None = None
    judge_context: str = ""

    @model_validator(mode="after")
    def _check_type_fields(self) -> "KPISpec":
        if self.type == "ordinal" and self.scale is None:
            raise ValueError(f"KPI {self.name}: tipo ordinal exige 'scale'")
        if self.type == "categorical" and not self.labels:
            raise ValueError(f"KPI {self.name}: tipo categorical exige 'labels'")
        return self

    @property
    def primary_metric(self) -> str:
        return {"ordinal": "qwk", "categorical": "macro_f1", "text": "judge_score"}[self.type]


# --------------------------------------------------------------------------- benchmark
class RunCfg(BaseModel):
    name: str = "benchmark"
    output_dir: str = "outputs"
    seed: int = 42
    max_workers: int = 8
    max_retries: int = 4
    cache_path: str | None = ".cache/llm_cache.sqlite"
    invalid_policy: Literal["pessimistic", "drop"] = "pessimistic"
    mock: bool = False


class DataCfg(BaseModel):
    path: str
    id_column: str = "call_id"
    summary_column: str = "resumo"
    transcript_column: str | None = "transcricao"
    reference_summary_column: str | None = "resumo_referencia"
    baseline_prefix: str | None = None
    sample: int | None = None


class JudgeCfg(BaseModel):
    enabled: bool = True
    model: str | None = None


class SummaryMetricsCfg(BaseModel):
    efficient: list[str] = Field(default_factory=list)
    modern: list[str] = Field(default_factory=list)
    bertscore_model: str = "xlm-roberta-large"
    embedding_model: str = "intfloat/multilingual-e5-base"
    nli_model: str = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
    claim_judge_model: str | None = None
    device: str = "cpu"


class StatsCfg(BaseModel):
    n_bootstrap: int = 2000
    ci: float = 0.95


class ProjectionCfg(BaseModel):
    monthly_calls: int = 1_000_000
    cache_hit_ratio: float = Field(0.9, ge=0, le=1)
    use_batch: bool = True


class BenchmarkConfig(BaseModel):
    run: RunCfg = RunCfg()
    data: DataCfg
    models: list[str]
    kpis: list[str] | Literal["all"] = "all"
    judge: JudgeCfg = JudgeCfg()
    summary_metrics: SummaryMetricsCfg = SummaryMetricsCfg()
    statistics: StatsCfg = StatsCfg()
    projection: ProjectionCfg = ProjectionCfg()
    simulation: dict[str, Any] = Field(default_factory=dict)

    # preenchidos em load_config
    model_registry: dict[str, ModelSpec] = Field(default_factory=dict)
    kpi_registry: dict[str, KPISpec] = Field(default_factory=dict)
    root: Path = PROJECT_ROOT

    @field_validator("models")
    @classmethod
    def _non_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("lista de modelos vazia")
        return v

    # ------------------------------------------------------------------ helpers
    @property
    def selected_kpis(self) -> list[KPISpec]:
        names = list(self.kpi_registry) if self.kpis == "all" else self.kpis
        return [self.kpi_registry[n] for n in names]

    @property
    def selected_models(self) -> list[ModelSpec]:
        return [self.model_registry[m] for m in self.models]

    def resolve(self, rel: str | None) -> Path | None:
        if rel is None:
            return None
        p = Path(rel)
        return p if p.is_absolute() else self.root / p

    def model(self, model_id: str) -> ModelSpec:
        if model_id not in self.model_registry:
            raise KeyError(f"modelo '{model_id}' não está em config/models.yaml")
        return self.model_registry[model_id]


def _read_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_config(path: str | Path, root: Path | None = None) -> BenchmarkConfig:
    """Lê o YAML do benchmark e anexa os registros de modelos e KPIs."""
    path = Path(path)
    root = root or (path.resolve().parent.parent if path.resolve().parent.name == "config" else PROJECT_ROOT)
    raw = _read_yaml(path)
    cfg = BenchmarkConfig(**raw, root=root)

    models_raw = _read_yaml(root / "config" / "models.yaml")["models"]
    cfg.model_registry = {k: ModelSpec(id=k, **v) for k, v in models_raw.items()}

    kpis_raw = _read_yaml(root / "config" / "kpis.yaml")["kpis"]
    cfg.kpi_registry = {k: KPISpec(name=k, **v) for k, v in kpis_raw.items()}

    # validações cruzadas
    unknown = [m for m in cfg.models if m not in cfg.model_registry]
    if unknown:
        raise ValueError(f"modelos não registrados em models.yaml: {unknown}")
    if cfg.kpis != "all":
        bad = [k for k in cfg.kpis if k not in cfg.kpi_registry]
        if bad:
            raise ValueError(f"KPIs não registrados em kpis.yaml: {bad}")
    for ref in (cfg.judge.model if cfg.judge.enabled else None, cfg.summary_metrics.claim_judge_model):
        if ref and ref not in cfg.model_registry:
            raise ValueError(f"modelo juiz '{ref}' não está em models.yaml")
    return cfg
