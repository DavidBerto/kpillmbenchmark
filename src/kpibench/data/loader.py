"""Leitura e validação do arquivo tabular de entrada (resumos + rótulos já validados)."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from kpibench.config import BenchmarkConfig, KPISpec

INVALID = "__invalido__"


def normalize_label(value) -> str | None:
    """'Não ' -> 'nao'; 'Cobrança Indevida' -> 'cobranca_indevida'."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    s = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    s = re.sub(r"[\s\-]+", "_", s.strip().lower())
    return s or None


def to_ordinal(value, kpi: KPISpec) -> int | None:
    try:
        v = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    assert kpi.scale is not None
    return v if kpi.scale.min <= v <= kpi.scale.max else None


@dataclass
class Dataset:
    df: pd.DataFrame
    id_col: str
    summary_col: str
    transcript_col: str | None
    reference_col: str | None
    baseline_prefix: str | None
    warnings: list[str] = field(default_factory=list)
    _index: dict | None = field(default=None, repr=False)

    @property
    def has_transcript(self) -> bool:
        return self.transcript_col is not None

    @property
    def has_reference(self) -> bool:
        return self.reference_col is not None

    def ids(self) -> list[str]:
        return self.df[self.id_col].astype(str).tolist()

    def record(self, call_id: str) -> dict:
        if self._index is None:
            self._index = {str(r[self.id_col]): r for r in self.df.to_dict(orient="records")}
        return self._index[str(call_id)]


def _read_any(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf == ".csv":
        return pd.read_csv(path, dtype={0: str})
    if suf in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if suf == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"formato não suportado: {suf} (use .csv, .xlsx ou .parquet)")


def load_dataset(cfg: BenchmarkConfig) -> Dataset:
    path = cfg.resolve(cfg.data.path)
    if path is None or not path.exists():
        raise FileNotFoundError(f"arquivo de entrada não encontrado: {path}")
    df = _read_any(path)
    d = cfg.data
    warnings: list[str] = []

    for col in (d.id_column, d.summary_column):
        if col not in df.columns:
            raise ValueError(f"coluna obrigatória ausente: '{col}'")
    if df[d.id_column].duplicated().any():
        dups = df.loc[df[d.id_column].duplicated(), d.id_column].tolist()[:5]
        raise ValueError(f"call_id duplicado: {dups}")
    df[d.id_column] = df[d.id_column].astype(str)

    empty = df[d.summary_column].isna() | (df[d.summary_column].astype(str).str.strip() == "")
    if empty.any():
        warnings.append(f"{int(empty.sum())} linhas sem resumo foram removidas")
        df = df.loc[~empty].copy()

    transcript_col = d.transcript_column if d.transcript_column in df.columns else None
    if d.transcript_column and transcript_col is None:
        warnings.append("coluna de transcrição ausente: métricas de fidelidade com fonte serão puladas")
    reference_col = d.reference_summary_column if d.reference_summary_column in df.columns else None
    if d.reference_summary_column and reference_col is None:
        warnings.append("resumo de referência ausente: métricas com referência usarão a transcrição como pseudo-referência")

    # normaliza rótulos ground-truth e verifica cobertura por KPI
    for kpi in cfg.selected_kpis:
        if kpi.gt_column not in df.columns:
            warnings.append(f"KPI '{kpi.name}': coluna '{kpi.gt_column}' ausente — KPI será pulado na avaliação")
            continue
        df[kpi.gt_column] = _normalize_series(df[kpi.gt_column], kpi)
        n_missing = int(df[kpi.gt_column].isna().sum())
        if n_missing:
            warnings.append(f"KPI '{kpi.name}': {n_missing} rótulos ausentes/fora do domínio (linhas ignoradas nesse KPI)")
        if kpi.type == "categorical":
            unknown = set(df[kpi.gt_column].dropna()) - set(kpi.labels or {})
            if unknown:
                warnings.append(f"KPI '{kpi.name}': rótulos fora da taxonomia {sorted(unknown)}")
        if d.baseline_prefix:
            bcol = f"{d.baseline_prefix}{kpi.name}"
            if bcol in df.columns and kpi.type != "text":
                df[bcol] = _normalize_series(df[bcol], kpi)

    if d.sample and d.sample < len(df):
        df = df.sample(n=d.sample, random_state=cfg.run.seed).reset_index(drop=True)

    return Dataset(df=df.reset_index(drop=True), id_col=d.id_column, summary_col=d.summary_column,
                   transcript_col=transcript_col, reference_col=reference_col,
                   baseline_prefix=d.baseline_prefix, warnings=warnings)


def _normalize_series(s: pd.Series, kpi: KPISpec) -> pd.Series:
    if kpi.type == "ordinal":
        return s.map(lambda v: to_ordinal(v, kpi)).astype("Int64")
    if kpi.type == "categorical":
        return s.map(normalize_label)
    return s.map(lambda v: None if pd.isna(v) or not str(v).strip() else str(v).strip())
