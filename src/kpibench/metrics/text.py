"""Métricas lexicais implementadas sem dependências (comportamento equivalente a
rouge-score sem stemming e sacrebleu sentence-level).

Todas retornam valores em [0, 1].
"""
from __future__ import annotations

import math
import re
from collections import Counter

_WORD = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _WORD.findall((text or "").lower())


def ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def _prf(overlap: float, n_cand: float, n_ref: float) -> tuple[float, float, float]:
    p = overlap / n_cand if n_cand else 0.0
    r = overlap / n_ref if n_ref else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


# --------------------------------------------------------------------------- ROUGE
def rouge_n(cand: str, ref: str, n: int = 1, return_all: bool = False):
    c, r = ngrams(tokenize(cand), n), ngrams(tokenize(ref), n)
    overlap = sum((c & r).values())
    prf = _prf(overlap, sum(c.values()), sum(r.values()))
    return prf if return_all else prf[2]


def _lcs(a: list[str], b: list[str]) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def rouge_l(cand: str, ref: str, return_all: bool = False):
    c, r = tokenize(cand), tokenize(ref)
    prf = _prf(_lcs(c, r), len(c), len(r))
    return prf if return_all else prf[2]


# --------------------------------------------------------------------------- BLEU
def bleu(cand: str, ref: str, max_n: int = 4) -> float:
    """BLEU sentence-level com smoothing 'add-1' nos n-gramas de ordem > 1 (Lin & Och, 2004)."""
    c, r = tokenize(cand), tokenize(ref)
    if not c or not r:
        return 0.0
    log_p = 0.0
    for n in range(1, max_n + 1):
        cn, rn = ngrams(c, n), ngrams(r, n)
        total = max(sum(cn.values()), 0)
        match = sum((cn & rn).values())
        if n > 1:
            match, total = match + 1, total + 1
        if total == 0 or match == 0:
            return 0.0
        log_p += math.log(match / total) / max_n
    bp = 1.0 if len(c) > len(r) else math.exp(1 - len(r) / len(c))
    return bp * math.exp(log_p)


# --------------------------------------------------------------------------- chrF / chrF++
def _char_ngrams(text: str, n: int) -> Counter:
    s = re.sub(r"\s+", "", (text or "").lower())
    return Counter(s[i:i + n] for i in range(len(s) - n + 1))


def chrf(cand: str, ref: str, char_order: int = 6, word_order: int = 0, beta: float = 2.0) -> float:
    """chrF (word_order=0) ou chrF++ (word_order=2), média macro sobre as ordens."""
    precs, recs = [], []
    for n in range(1, char_order + 1):
        cn, rn = _char_ngrams(cand, n), _char_ngrams(ref, n)
        if not rn or not cn:
            continue
        m = sum((cn & rn).values())
        precs.append(m / sum(cn.values()))
        recs.append(m / sum(rn.values()))
    ct, rt = tokenize(cand), tokenize(ref)
    for n in range(1, word_order + 1):
        cn, rn = ngrams(ct, n), ngrams(rt, n)
        if not rn or not cn:
            continue
        m = sum((cn & rn).values())
        precs.append(m / sum(cn.values()))
        recs.append(m / sum(rn.values()))
    if not precs:
        return 0.0
    p, r = sum(precs) / len(precs), sum(recs) / len(recs)
    if p + r == 0:
        return 0.0
    b2 = beta ** 2
    return (1 + b2) * p * r / (b2 * p + r)


# --------------------------------------------------------------------------- TF-IDF
def tfidf_cosine_pairs(cands: list[str], refs: list[str]) -> list[float]:
    """Cosseno TF-IDF (uni+bigramas) ajustado no corpus de candidatos + referências."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, lowercase=True, token_pattern=r"(?u)\b\w+\b")
    vec.fit([*cands, *refs])
    a = normalize(vec.transform(cands))
    b = normalize(vec.transform(refs))
    return [float(x) for x in a.multiply(b).sum(axis=1).A1]
