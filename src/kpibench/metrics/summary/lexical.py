"""Métricas EFICIENTES (CPU, custo ~zero): sobreposição lexical e heurísticas factuais."""
from __future__ import annotations

import re
import time

import numpy as np

from kpibench.metrics import text as T
from kpibench.metrics.summary.base import MetricResult


def _timed(fn, pairs):
    t0 = time.perf_counter()
    vals = np.array([fn(a, b) for a, b in pairs], dtype=float)
    ms = (time.perf_counter() - t0) * 1000 / max(len(pairs), 1)
    return vals, ms


def reference_metrics(summaries: list[str], refs: list[str], ref_kind: str, wanted: set[str]) -> list[MetricResult]:
    pairs = list(zip(summaries, refs))
    out: list[MetricResult] = []
    if "rouge" in wanted:
        for name, fn in (("rouge1_f", lambda a, b: T.rouge_n(a, b, 1)),
                         ("rouge2_f", lambda a, b: T.rouge_n(a, b, 2)),
                         ("rougeL_f", T.rouge_l)):
            v, ms = _timed(fn, pairs)
            out.append(MetricResult(name, "lexical", "eficiente", ref_kind, v, ms_per_item=ms))
    if "bleu" in wanted:
        v, ms = _timed(T.bleu, pairs)
        out.append(MetricResult("bleu", "lexical", "eficiente", ref_kind, v, ms_per_item=ms))
    if "chrf" in wanted:
        v, ms = _timed(lambda a, b: T.chrf(a, b, word_order=2), pairs)
        out.append(MetricResult("chrf++", "lexical", "eficiente", ref_kind, v, ms_per_item=ms))
    if "tfidf_cosine" in wanted:
        t0 = time.perf_counter()
        v = np.array(T.tfidf_cosine_pairs(summaries, refs))
        ms = (time.perf_counter() - t0) * 1000 / max(len(pairs), 1)
        out.append(MetricResult("tfidf_cosine", "lexical", "eficiente", ref_kind, v, ms_per_item=ms))
    return out


# --------------------------------------------------------------------------- fonte (transcrição)
def source_metrics(summaries: list[str], sources: list[str], wanted: set[str]) -> list[MetricResult]:
    pairs = list(zip(summaries, sources))
    out: list[MetricResult] = []
    if "source_rouge" in wanted:
        # PRECISÃO contra a fonte: fração dos n-gramas do resumo presentes na transcrição
        v, ms = _timed(lambda a, b: T.rouge_n(a, b, 2, return_all=True)[0], pairs)
        out.append(MetricResult("source_rouge2_precision", "factual_heuristica", "eficiente", "fonte", v,
                                ms_per_item=ms, note="proxy barato de fidelidade (extratividade)"))
    if "novel_ngrams" in wanted:
        v, ms = _timed(novel_bigram_rate, pairs)
        out.append(MetricResult("novel_bigram_rate", "factual_heuristica", "eficiente", "fonte", v,
                                direction=-1, ms_per_item=ms,
                                note="bigramas do resumo ausentes da fonte; menor é melhor"))
    if "numeric_consistency" in wanted:
        v, ms = _timed(numeric_consistency, pairs)
        out.append(MetricResult("numeric_consistency", "factual_heuristica", "eficiente", "fonte", v,
                                ms_per_item=ms,
                                note="valores/datas/protocolos do resumo presentes na transcrição"))
    return out


def novel_bigram_rate(summary: str, source: str) -> float:
    s = T.ngrams(T.tokenize(summary), 2)
    if not s:
        return 0.0
    src = set(T.ngrams(T.tokenize(source), 2))
    total = sum(s.values())
    return sum(c for g, c in s.items() if g not in src) / total


_NUM = re.compile(r"\d+(?:[.,/:-]\d+)*")


def _nums(text: str) -> set[str]:
    out = set()
    for m in _NUM.findall(text or ""):
        m = m.strip(".,:-/")
        if not m:
            continue
        canon = re.sub(r"[.](?=\d{3}\b)", "", m)  # 1.234,56 -> 1234,56
        out.add(canon.replace(".", ","))
    return out


def numeric_consistency(summary: str, source: str) -> float:
    """1.0 se todos os números do resumo aparecem na transcrição; resumo sem números -> 1.0.
    Limitação: números por extenso na transcrição (ASR) não são reconhecidos."""
    s = _nums(summary)
    if not s:
        return 1.0
    src = _nums(source)
    return len(s & src) / len(s)
