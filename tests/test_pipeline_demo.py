"""Smoke test end-to-end com o provider simulado (sem rede)."""
import sys

import pandas as pd

from kpibench.config import PROJECT_ROOT, load_config
from kpibench.pipeline import run

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from generate_sample_data import main as generate  # noqa: E402


def test_demo_pipeline_end_to_end(tmp_path):
    data = generate(n=40, out=tmp_path / "amostra.csv")
    cfg = load_config(PROJECT_ROOT / "config" / "demo.yaml")
    cfg.data.path = str(data)
    cfg.run.output_dir = str(tmp_path / "out")
    cfg.statistics.n_bootstrap = 100
    cfg.models = ["claude-haiku-5-5", "gpt-4o-mini"]

    out = run(cfg)

    assert (out / "report.md").exists()
    rec = pd.read_csv(out / "recomendacao_roteamento.csv")
    assert set(rec["kpi"]) == set(cfg.kpi_registry)
    preds = pd.read_csv(out / "predictions.csv")
    assert len(preds) == 40 * 2 * len(cfg.kpi_registry)
    # um único hash de prompt por KPI, compartilhado por todos os modelos
    assert preds.groupby("kpi")["prompt_hash"].nunique().max() == 1
    meta = pd.read_csv(out / "meta_avaliacao_metricas_resumo.csv")
    assert {"rougeL_f", "numeric_consistency"} <= set(meta["metric"])
    full = pd.read_csv(out / "metricas_por_kpi_modelo.csv")
    assert "producao_atual" in set(full["model"])
    # custo-efetividade: ICER na fronteira, por KPI e agregado
    ce = pd.read_csv(out / "custo_efetividade_por_kpi.csv")
    assert set(ce["kpi"]) == set(cfg.kpi_registry) and "producao_atual" not in set(ce["model"])
    assert ce["q_norm"].between(0, 1).all() and ce["p_fronteira"].between(0, 1).all()
    assert (ce.groupby("kpi")["status_ce"].apply(lambda s: (s == "fronteira").any())).all()
    agg = pd.read_csv(out / "custo_efetividade_agregado.csv")
    assert set(agg["model"]) == set(cfg.models)
    assert {"status_ce", "icer_usd_1k_por_pp"} <= set(full.columns) and "perf_por_usd" not in full.columns
    assert "Custo-efetividade" in (out / "report.md").read_text(encoding="utf-8")
    assert (out / "icer_fronteira.png").exists()
