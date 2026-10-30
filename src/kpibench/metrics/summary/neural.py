"""Métricas MODERNAS baseadas em modelos locais (dependências opcionais: '.[neural]').

- BERTScore (Zhang et al., 2020): similaridade contextual token-a-token com a referência.
- Cosseno de embeddings (multilingual-e5): similaridade semântica global.
- SummaC-ZS (Laban et al., 2022): fidelidade via NLI; para cada frase do resumo pega a
  maior pontuação (P_entail - P_contradiction) sobre janelas da transcrição e tira a média.

Se a dependência ou o download do modelo falhar, a métrica é marcada como pulada
(com o motivo) e o restante da suíte segue.
"""
from __future__ import annotations

import re
import time

import numpy as np

from kpibench.metrics.summary.base import MetricResult, skipped


def bertscore(summaries: list[str], refs: list[str], ref_kind: str, model: str, device: str) -> MetricResult:
    n = len(summaries)
    try:
        from bert_score import score  # type: ignore

        t0 = time.perf_counter()
        _, _, f1 = score(summaries, refs, model_type=model, device=device, lang="pt", verbose=False)
        ms = (time.perf_counter() - t0) * 1000 / max(n, 1)
        return MetricResult("bertscore_f1", "embedding", "moderna", ref_kind, f1.numpy().astype(float),
                            compute="GPU/CPU", ms_per_item=ms, note=model)
    except Exception as e:  # noqa: BLE001
        return skipped("bertscore_f1", "embedding", "moderna", ref_kind, n, f"{type(e).__name__}: {e}"[:160])


def embedding_cosine(summaries: list[str], refs: list[str], ref_kind: str, model: str, device: str) -> MetricResult:
    n = len(summaries)
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        m = SentenceTransformer(model, device=device)
        prefix = "query: " if "e5" in model else ""
        t0 = time.perf_counter()
        a = m.encode([prefix + s for s in summaries], normalize_embeddings=True)
        b = m.encode([prefix + r for r in refs], normalize_embeddings=True)
        ms = (time.perf_counter() - t0) * 1000 / max(n, 1)
        return MetricResult("embedding_cosine", "embedding", "moderna", ref_kind,
                            np.sum(a * b, axis=1).astype(float), compute="GPU/CPU", ms_per_item=ms, note=model)
    except Exception as e:  # noqa: BLE001
        return skipped("embedding_cosine", "embedding", "moderna", ref_kind, n, f"{type(e).__name__}: {e}"[:160])


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?;])\s+", text or "") if len(s.strip()) > 3]


def _windows(transcript: str, size: int = 3) -> list[str]:
    turns = [t.strip() for t in (transcript or "").splitlines() if t.strip()] or _sentences(transcript)
    if len(turns) <= size:
        return [" ".join(turns)]
    return [" ".join(turns[i:i + size]) for i in range(0, len(turns) - size + 1, max(1, size - 1))]


def summac_zs(summaries: list[str], sources: list[str], model: str, device: str) -> MetricResult:
    n = len(summaries)
    try:
        import torch  # type: ignore
        from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore

        tok = AutoTokenizer.from_pretrained(model)
        nli = AutoModelForSequenceClassification.from_pretrained(model).to(device).eval()
        id2label = {i: l.lower() for i, l in nli.config.id2label.items()}
        ent = next(i for i, l in id2label.items() if "entail" in l)
        con = next(i for i, l in id2label.items() if "contra" in l)

        t0 = time.perf_counter()
        vals = []
        for summ, src in zip(summaries, sources):
            sents, wins = _sentences(summ), _windows(src)
            if not sents:
                vals.append(np.nan)
                continue
            prem = [w for w in wins for _ in sents]
            hyp = [s for _ in wins for s in sents]
            scores = []
            with torch.no_grad():
                for i in range(0, len(prem), 32):
                    enc = tok(prem[i:i + 32], hyp[i:i + 32], truncation="only_first", max_length=512,
                              padding=True, return_tensors="pt").to(device)
                    p = torch.softmax(nli(**enc).logits, dim=-1)
                    scores.append((p[:, ent] - p[:, con]).cpu().numpy())
            mat = np.concatenate(scores).reshape(len(wins), len(sents))
            vals.append(float(mat.max(axis=0).mean()))
        ms = (time.perf_counter() - t0) * 1000 / max(n, 1)
        return MetricResult("summac_zs", "nli", "moderna", "fonte", np.array(vals, float),
                            compute="GPU/CPU", ms_per_item=ms, note=model)
    except Exception as e:  # noqa: BLE001
        return skipped("summac_zs", "nli", "moderna", "fonte", n, f"{type(e).__name__}: {e}"[:160])
