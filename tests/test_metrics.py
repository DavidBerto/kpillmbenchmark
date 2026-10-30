import numpy as np
import pytest
from sklearn.metrics import cohen_kappa_score, f1_score

from kpibench.metrics import classification as C
from kpibench.metrics import text as T
from kpibench.metrics.stats import compare_to_reference
from kpibench.metrics.summary.lexical import novel_bigram_rate, numeric_consistency


def test_qwk_matches_sklearn():
    rng = np.random.default_rng(1)
    for _ in range(50):
        a = rng.integers(1, 6, 80)
        b = np.clip(a + rng.integers(-2, 3, 80), 1, 5)
        ref = cohen_kappa_score(a, b, weights="quadratic", labels=[1, 2, 3, 4, 5])
        assert C.qwk(a, b, 1, 5) == pytest.approx(ref, abs=1e-9)


def test_qwk_perfect_and_constant():
    assert C.qwk([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], 1, 5) == pytest.approx(1.0)
    assert np.isnan(C.qwk([3, 3, 3], [3, 3, 3], 1, 5))


def test_macro_f1_invalid_counts_as_error():
    yt = ["sim", "nao", "sim", "parcial"]
    assert C.macro_f1(yt, yt, ["sim", "nao", "parcial"]) == 1.0
    yp = ["sim", "__invalido__", "sim", "parcial"]
    ref = f1_score(yt, yp, labels=["sim", "nao", "parcial"], average="macro", zero_division=0)
    assert C.macro_f1(yt, yp, ["sim", "nao", "parcial"]) == pytest.approx(ref)
    assert C.macro_f1(yt, yp, ["sim", "nao", "parcial"]) < 1.0


def test_lexical_identity_and_disjoint():
    s = "Cliente contestou cobrança de R$ 29,90 do serviço Música Plus."
    assert T.rouge_n(s, s, 1) == pytest.approx(1.0)
    assert T.rouge_l(s, s) == pytest.approx(1.0)
    assert T.bleu(s, s) == pytest.approx(1.0)
    assert T.chrf(s, s, word_order=2) == pytest.approx(1.0)
    assert T.rouge_n("abc def", "xyz uvw", 1) == 0.0


def test_rouge_l_known_value():
    # LCS("a b c d", "a c d e") = 3 -> P=R=0.75 -> F=0.75
    assert T.rouge_l("a b c d", "a c d e") == pytest.approx(0.75)


def test_numeric_consistency_flags_hallucinated_values():
    src = "ATENDENTE: o estorno de R$ 52,90 sai em 5 dias úteis, protocolo 12345678901."
    assert numeric_consistency("Estorno de R$ 52,90 em 5 dias úteis.", src) == 1.0
    assert numeric_consistency("Estorno de R$ 99,90 em 5 dias úteis.", src) == pytest.approx(0.5)
    assert numeric_consistency("Cliente foi bem atendido.", src) == 1.0


def test_novel_bigram_rate_bounds():
    src = "o cliente pediu segunda via da fatura"
    assert novel_bigram_rate("cliente pediu segunda via", src) == 0.0
    assert novel_bigram_rate("cancelamento imediato do plano", src) == 1.0


def test_non_inferiority_decisions():
    rng = np.random.default_rng(0)
    ref = rng.normal(0.90, 0.005, 2000)
    same = ref - rng.normal(0.0, 0.003, 2000)
    worse = ref - 0.10
    noisy = ref - rng.normal(0.03, 0.05, 2000)
    point = {"ref": 0.90, "same": 0.90, "worse": 0.80, "noisy": 0.87}
    out = {c.model: c.decision for c in compare_to_reference(
        point, {"ref": ref, "same": same, "worse": worse, "noisy": noisy}, "ref", margin=0.02, ci=0.95)}
    assert out == {"ref": "referencia", "same": "nao_inferior", "worse": "inferior", "noisy": "inconclusivo"}
