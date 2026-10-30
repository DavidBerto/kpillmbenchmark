"""Avaliação por KPI: métrica primária com IC bootstrap, secundárias e
distribuições bootstrap pareadas (insumo do roteamento custo x performance)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from kpibench.config import BenchmarkConfig, KPISpec
from kpibench.data.loader import INVALID, Dataset
from kpibench.metrics import classification as C
from kpibench.metrics import text as T
from kpibench.metrics.stats import bootstrap_dist, bootstrap_indices, percentile_ci

BASELINE_NAME = "producao_atual"


@dataclass
class KPIEvaluation:
    kpi: KPISpec
    primary_name: str
    n_items: int
    table: pd.DataFrame                      # uma linha por modelo
    dists: dict[str, np.ndarray]             # distribuição bootstrap da primária por modelo
    per_class: pd.DataFrame | None = None    # categóricos
    confusion: dict[str, pd.DataFrame] = field(default_factory=dict)


def _pessimistic_ordinal(gt: int, lo: int, hi: int) -> int:
    return hi if gt <= (lo + hi) / 2 else lo


def evaluate_kpi(cfg: BenchmarkConfig, ds: Dataset, kpi: KPISpec, preds: pd.DataFrame,
                 judge: pd.DataFrame | None) -> KPIEvaluation | None:
    if kpi.gt_column not in ds.df.columns:
        return None
    gt = ds.df[[ds.id_col, kpi.gt_column]].dropna()
    gt = gt.rename(columns={ds.id_col: "call_id", kpi.gt_column: "gt"})
    ids = gt["call_id"].astype(str).tolist()
    n = len(ids)
    if n < 2:
        return None

    # ----- predições alinhadas por modelo (inclui baseline de produção se houver)
    sub = preds[preds["kpi"] == kpi.name]
    model_frames: dict[str, pd.DataFrame] = {}
    for m, g in sub.groupby("model"):
        model_frames[m] = gt.merge(g, on="call_id", how="left")
    if ds.baseline_prefix and f"{ds.baseline_prefix}{kpi.name}" in ds.df.columns:
        bcol = f"{ds.baseline_prefix}{kpi.name}"
        b = ds.df[[ds.id_col, bcol]].rename(columns={ds.id_col: "call_id", bcol: "value"})
        b["valid"] = b["value"].notna()
        if kpi.type == "categorical":
            b["valid"] &= b["value"].isin(list(kpi.labels or {}))
        model_frames[BASELINE_NAME] = gt.merge(b, on="call_id", how="left")

    idx = bootstrap_indices(n, cfg.statistics.n_bootstrap, cfg.run.seed)
    pessimistic = cfg.run.invalid_policy == "pessimistic"
    rows, dists, per_class_rows, confusion = [], {}, [], {}

    judge_on = kpi.type == "text" and judge is not None and not judge.empty and cfg.judge.enabled
    primary_name = kpi.primary_metric if (kpi.type != "text" or judge_on) else "chrf"

    for m, f in model_frames.items():
        valid = f["valid"].fillna(False).astype(bool).to_numpy()
        row = {"kpi": kpi.name, "model": m, "n": n, "valid_rate": float(valid.mean())}

        if kpi.type == "ordinal":
            lo, hi = kpi.scale.min, kpi.scale.max
            yt = f["gt"].astype(int).to_numpy()
            yp_raw = pd.to_numeric(f["value"], errors="coerce").to_numpy()
            yp = np.array([int(p) if v else _pessimistic_ordinal(t, lo, hi)
                           for p, v, t in zip(yp_raw, valid, yt)])
            mask = np.ones(n, bool) if pessimistic else valid

            def fn(ix, yt=yt, yp=yp, mask=mask):
                ix = ix[mask[ix]]
                return C.qwk(yt[ix], yp[ix], lo, hi)
            sel = mask
            row.update(qwk=C.qwk(yt[sel], yp[sel], lo, hi), spearman=C.spearman(yt[sel], yp[sel]),
                       mae=C.mae(yt[sel], yp[sel]), exact_acc=C.exact_acc(yt[sel], yp[sel]),
                       within1_acc=C.within1_acc(yt[sel], yp[sel]))

        elif kpi.type == "categorical":
            labels = list(kpi.labels)
            yt = f["gt"].astype(str).to_numpy()
            yp = np.array([str(p) if v else INVALID for p, v in zip(f["value"], valid)])
            mask = np.ones(n, bool) if pessimistic else valid

            def fn(ix, yt=yt, yp=yp, mask=mask):
                ix = ix[mask[ix]]
                return C.macro_f1(yt[ix], yp[ix], labels)
            sel = mask
            row.update(macro_f1=C.macro_f1(yt[sel], yp[sel], labels), accuracy=C.exact_acc(yt[sel], yp[sel]),
                       kappa=C.cohen_kappa(yt[sel], yp[sel]))
            for lab, (f1, sup) in C.per_class_f1(yt[sel], yp[sel], labels).items():
                per_class_rows.append({"kpi": kpi.name, "model": m, "label": lab, "f1": f1, "support": sup})
            confusion[m] = pd.crosstab(pd.Series(yt[sel], name="gt"), pd.Series(yp[sel], name="pred"))

        else:  # text
            refs = f["gt"].astype(str).tolist()
            cands = [str(p) if v else "" for p, v in zip(f["value"], valid)]
            chrf_s = np.array([T.chrf(c, r, word_order=2) if c else 0.0 for c, r in zip(cands, refs)])
            rl = np.array([T.rouge_l(c, r) if c else 0.0 for c, r in zip(cands, refs)])
            tf = np.array(T.tfidf_cosine_pairs(cands, refs))
            js = np.full(n, np.nan)
            if judge_on:
                jm = judge[(judge["kpi"] == kpi.name) & (judge["model"] == m)].set_index("call_id")["judge_score"]
                js = np.array([jm.get(cid, np.nan) for cid in f["call_id"]], dtype=float)
                judged = np.isfinite(js)
                row["judge_coverage"] = float(judged[valid].mean()) if valid.any() else 0.0
                js = np.where(valid, js, 0.0 if pessimistic else np.nan)
            score = js if judge_on else chrf_s
            if pessimistic:
                score = np.where(valid, score, 0.0)
            mask = np.isfinite(score) & (np.ones(n, bool) if pessimistic else valid)

            def fn(ix, score=score, mask=mask):
                ix = ix[mask[ix]]
                return float(np.mean(score[ix])) if len(ix) else float("nan")
            row.update(chrf=float(chrf_s[valid].mean()) if valid.any() else 0.0,
                       rouge_l=float(rl[valid].mean()) if valid.any() else 0.0,
                       tfidf_cos=float(tf[valid].mean()) if valid.any() else 0.0)
            if judge_on:
                row["judge_score"] = fn(np.arange(n))

        point = fn(np.arange(n))
        dist = bootstrap_dist(fn, idx)
        lo_ci, hi_ci = percentile_ci(dist, cfg.statistics.ci)
        row.update(primary_metric=primary_name, primary=point, ci_low=lo_ci, ci_high=hi_ci)
        rows.append(row)
        dists[m] = dist

    table = pd.DataFrame(rows).sort_values("primary", ascending=False).reset_index(drop=True)
    per_class = pd.DataFrame(per_class_rows) if per_class_rows else None
    return KPIEvaluation(kpi, primary_name, n, table, dists, per_class, confusion)
