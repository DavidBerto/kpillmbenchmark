"""Teste de modelos em ligações reais via Nexus (usado pelo notebook 03).

Fluxo:
  ligações (Azure, DefaultAzureCredential)            -> kpibench.data.azure_calls.load_calls
  referência (modelo via Nexus ou arquivo validado)   -> gt_<kpi>
  baseline de produção (só KPIs de domínio compatível) -> prod_<kpi>
  candidatos via Nexus + avaliação estatística          -> kpibench.pipeline.run
  tabelas de negócio (placar por modelo, resumo executivo, drill-down por ligação)

Sobre a referência: a base de produção não tem rótulos auditados, e vários KPIs de
produção estão num domínio diferente do benchmark (Sim/Não onde o benchmark usa nota
1–5 ou taxonomia). Por isso o padrão é usar um MODELO DE REFERÊNCIA (o mais forte
disponível no Nexus) como pseudo-rótulo: as métricas medem CONCORDÂNCIA COM A
REFERÊNCIA, não acerto. Com rótulos auditados, passe um arquivo (`reference="arquivo:..."`).
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from kpibench.config import PROJECT_ROOT, BenchmarkConfig, load_config
from kpibench.data.loader import load_dataset, normalize_label

BASELINE_NAME = "producao_atual"
SAMPLE_PATH = PROJECT_ROOT / "data" / "input" / "exemplo_kpis_validacao.csv"

# Tokens de saída assumidos na estimativa (sem raciocínio), por tipo de KPI
OUT_TOKENS = {"ordinal": 60, "categorical": 60, "text": 70}
OUT_TOKENS_SUMMARY_QUALITY = 250
REASONING_OVERHEAD = 300


# =============================================================================== provedores
def provider_of(model: str) -> str:
    m = model.lower()
    if m == BASELINE_NAME:
        return "Produção atual"
    if m.startswith(("gpt", "o1", "o3", "o4")):
        return "OpenAI"
    if m.startswith("claude"):
        return "Anthropic"
    if m.startswith("gemini"):
        return "Google"
    if m.startswith("grok"):
        return "xAI"
    if m.startswith("deepseek"):
        return "DeepSeek"
    if m.startswith("kimi"):
        return "Moonshot"
    return "Outros"


# =============================================================================== configuração
def build_config(*, models: list[str], kpis: list[str] | str = "all", judge_model: str | None = None,
                 simulated: bool = False, n_bootstrap: int = 1000, max_workers: int = 8,
                 llm_summary_metrics: bool = False, base_config: str | Path | None = None) -> BenchmarkConfig:
    """Config da rodada a partir de config/benchmark.yaml (ou demo.yaml no modo simulado)."""
    base = Path(base_config) if base_config else PROJECT_ROOT / "config" / ("demo.yaml" if simulated else "benchmark.yaml")
    cfg = load_config(base)
    unknown = [m for m in models if m not in cfg.model_registry]
    if unknown:
        raise ValueError(f"modelos fora de config/models.yaml: {unknown}")
    cfg.models = list(models)
    cfg.kpis = kpis
    if judge_model:
        if judge_model not in cfg.model_registry:
            raise ValueError(f"juiz '{judge_model}' fora de config/models.yaml")
        cfg.judge.model = judge_model
        cfg.summary_metrics.claim_judge_model = judge_model
    cfg.statistics.n_bootstrap = n_bootstrap
    cfg.run.max_workers = max_workers
    cfg.run.mock = simulated
    # métricas de resumo: lexicais sempre; G-Eval é o próprio KPI (sem custo extra);
    # fidelidade por afirmações e modelos neurais só se pedido (custam chamadas/GPU)
    cfg.summary_metrics.modern = (["geval", "claim_faithfulness"] if llm_summary_metrics else ["geval"])
    return cfg


# =============================================================================== Nexus
def nexus_models() -> list[str]:
    """Ids expostos pelo gateway (GET /models), com as variáveis do projeto (base_url, bearer_token)."""
    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:
        pass
    import openai

    base_url, token = os.environ.get("base_url"), os.environ.get("bearer_token")
    if not base_url or not token:
        raise RuntimeError("defina base_url e bearer_token no .env para acessar o Nexus")
    client = openai.OpenAI(base_url=base_url, api_key=token)
    return sorted(m.id for m in client.models.list().data)


def check_models(cfg: BenchmarkConfig, models: list[str], available: list[str] | None) -> pd.DataFrame:
    rows = []
    for m in models:
        spec = cfg.model(m)
        rows.append({"modelo": m, "provedor": provider_of(m), "id_no_nexus": spec.model,
                     "disponivel_no_nexus": (spec.model in available) if available is not None else None,
                     "preco_entrada_usd_1M": spec.pricing.input, "preco_saida_usd_1M": spec.pricing.output,
                     "preco_estimado": spec.pricing_is_estimate})
    return pd.DataFrame(rows)


# =============================================================================== ligações
def sample_calls(num_calls: int, seed: int = 7) -> pd.DataFrame:
    """Modo simulado: ligações do arquivo de exemplo, no formato da carga do Azure."""
    df = pd.read_csv(SAMPLE_PATH, dtype={"call_id": str})
    n = max(10, min(num_calls or len(df), len(df)))
    df = df.sample(n=n, random_state=seed).reset_index(drop=True)
    df["data_referencia"] = pd.to_datetime(df["data_chamada"])
    # domínio BRUTO de produção, como vem do Azure: CSAT 0–10, resolvida/FCR Sim/Não
    rng = np.random.default_rng(seed)
    df["prod_csat"] = (df["csat"].astype(float) * 2 - rng.integers(0, 2, len(df))).clip(0, 10)
    df["prod_solicitacao_resolvida"] = df["solved"].replace({"parcial": "nao"})
    df["prod_first_call_resolution"] = df["dash_advins_fcr"].replace({"indeterminado": "nao"})
    df["prod_razao_csat"] = df["csat_reasoning"]
    df["prod_solver_reasoning"] = df["solved_reasoning"]
    df["prod_razao_ligacao"] = df["call_reason"]
    df["origem_resumo"] = "simulado"
    return df


# =============================================================================== baseline de produção
def csat_0_10_to_1_5(v) -> float:
    """Premissa: CSAT de produção em 0–10 ('Nota: 7') agrupado em 5 faixas iguais."""
    if v is None or pd.isna(v):
        return np.nan
    v = float(v)
    return float(1 if v <= 2 else 2 if v <= 4 else 3 if v <= 6 else 4 if v <= 8 else 5)


def map_production_baseline(calls: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """prod_<kpi> no domínio do benchmark, só para KPIs comparáveis. Retorna (colunas, notas)."""
    out = pd.DataFrame({"call_id": calls["call_id"].astype(str)})
    notes = []

    def add(kpi, series, regra):
        out[f"prod_{kpi}"] = series.to_numpy()
        notes.append({"kpi": kpi, "comparavel": True, "regra": regra})

    if "prod_csat" in calls:
        add("csat", calls["prod_csat"].map(csat_0_10_to_1_5), "0–10 → 1–5 (≤2→1, ≤4→2, ≤6→3, ≤8→4, >8→5)")
    if "prod_solicitacao_resolvida" in calls:
        add("solicitacao_resolvida", calls["prod_solicitacao_resolvida"].map(normalize_label),
            "Sim/Não; produção não expressa 'parcial' (conta como erro quando a referência diz parcial)")
    if "prod_first_call_resolution" in calls:
        add("first_call_resolution", calls["prod_first_call_resolution"].map(normalize_label),
            "Sim/Não; produção não expressa 'indeterminado'")
    for kpi in ("razao_csat", "solver_reasoning"):
        if f"prod_{kpi}" in calls:
            add(kpi, calls[f"prod_{kpi}"], "texto livre, avaliado pelo juiz contra a referência")
    for kpi, motivo in (("empatia", "produção Sim/Não × benchmark nota 1–5"),
                        ("clareza_informacoes", "produção Sim/Não × benchmark nota 1–5"),
                        ("odc", "produção Sim/Não × benchmark taxonomia de 7 códigos"),
                        ("razao_ligacao", "produção texto livre × benchmark 8 rótulos"),
                        ("qualidade_resumo", "não existe em produção")):
        notes.append({"kpi": kpi, "comparavel": False, "regra": motivo})
    return out, pd.DataFrame(notes)


# =============================================================================== estimativa de custo
def estimate_cost(cfg: BenchmarkConfig, calls: pd.DataFrame, reference_model: str | None) -> pd.DataFrame:
    """Custo estimado da rodada (antes de executar), por modelo, a partir dos prompts reais."""
    from kpibench.extraction.runner import kpi_prompt_vars
    from kpibench.llm.tokens import estimate_tokens
    from kpibench.prompts.renderer import PromptRenderer

    ds = _dataset_from_frame(cfg, calls[["call_id", "resumo"] + (["transcricao"] if "transcricao" in calls else [])],
                             PROJECT_ROOT / ".cache" / "estimate_dataset.csv")
    renderer = PromptRenderer(cfg.root / "prompts")
    tok = {}
    for kpi in cfg.selected_kpis:
        ins = [estimate_tokens(p.system) + estimate_tokens(p.user) for p in
               (renderer.render(f"kpis/{kpi.prompt}", **kpi_prompt_vars(kpi, ds, ds.record(c))) for c in ds.ids())]
        out = OUT_TOKENS_SUMMARY_QUALITY if kpi.name == "qualidade_resumo" else OUT_TOKENS[kpi.type]
        tok[kpi.name] = (float(np.mean(ins)), out)
    rows = []
    models = list(cfg.models) + ([reference_model] if reference_model and reference_model not in cfg.models else [])
    n = len(ds.ids())
    for m in models:
        spec = cfg.model(m)
        reasoning = REASONING_OVERHEAD if "reasoning_effort" in spec.params else 0
        usd = sum((i * spec.pricing.input + (o + reasoning) * spec.pricing.output) / 1e6 for i, o in tok.values()) * n
        rows.append({"modelo": m, "papel": "referência" if m == reference_model else "candidato",
                     "chamadas_llm": n * len(tok), "usd_estimado": usd, "preco_estimado": spec.pricing_is_estimate})
    judge_kpis = [k for k in cfg.selected_kpis if k.type == "text"]
    if cfg.judge.enabled and cfg.judge.model and judge_kpis:
        spec = cfg.model(cfg.judge.model)
        calls_judge = n * len(judge_kpis) * (len(cfg.models) + 1)
        usd = calls_judge * (700 * spec.pricing.input + 120 * spec.pricing.output) / 1e6
        rows.append({"modelo": cfg.judge.model, "papel": "juiz (texto livre)", "chamadas_llm": calls_judge,
                     "usd_estimado": usd, "preco_estimado": spec.pricing_is_estimate})
    df = pd.DataFrame(rows)
    df.loc[len(df)] = {"modelo": "TOTAL", "papel": "", "chamadas_llm": int(df["chamadas_llm"].sum()),
                       "usd_estimado": float(df["usd_estimado"].sum()), "preco_estimado": bool(df["preco_estimado"].any())}
    return df


# =============================================================================== referência
def _dataset_from_frame(cfg: BenchmarkConfig, frame: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    c = cfg.model_copy(deep=True)
    c.data.path = str(path)
    c.data.reference_summary_column = None
    c.data.baseline_prefix = None
    c.data.sample = None
    if "transcricao" not in frame.columns:
        c.data.transcript_column = None
    ds = load_dataset(c)
    ds.warnings.clear()
    return ds


def reference_labels(cfg: BenchmarkConfig, calls: pd.DataFrame, reference: str, out_dir: Path
                     ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """gt_<kpi> por ligação. `reference` = id de modelo (via Nexus) ou 'arquivo:<csv com call_id + gt_*>'."""
    kpis = cfg.selected_kpis
    if reference.startswith("arquivo:"):
        gt = pd.read_csv(reference.split(":", 1)[1], dtype={"call_id": str})
        cols = ["call_id"] + [k.gt_column for k in kpis if k.gt_column in gt.columns]
        return gt[cols], pd.DataFrame()

    from kpibench.extraction.runner import LLMExecutor, run_kpi_extraction
    from kpibench.prompts.renderer import PromptRenderer

    base_cols = ["call_id", "resumo"] + (["transcricao"] if "transcricao" in calls else [])
    # no modo simulado o mock precisa dos rótulos do arquivo de exemplo para gerar a "referência"
    extra = [k.gt_column for k in kpis if cfg.run.mock and k.gt_column in calls.columns]
    ds = _dataset_from_frame(cfg, calls[base_cols + extra], out_dir / "_entrada_referencia.csv")
    c = cfg.model_copy(deep=True)
    c.models = [reference]
    print(f"Referência: {reference} em {len(ds.ids())} ligações × {len(kpis)} KPIs", file=sys.stderr)
    preds = run_kpi_extraction(c, ds, PromptRenderer(cfg.root / "prompts"), LLMExecutor(c, ds))
    preds.to_csv(out_dir / "referencia_predicoes.csv", index=False)
    gt = pd.DataFrame({"call_id": ds.ids()})
    for k in kpis:
        sub = preds[(preds["kpi"] == k.name) & preds["valid"]].set_index("call_id")["value"]
        gt[k.gt_column] = gt["call_id"].map(sub)
    inval = preds.groupby("kpi")["valid"].mean()
    if (inval < 1).any():
        print("Referência com saídas inválidas (ligações ignoradas nesses KPIs):\n"
              + (1 - inval[inval < 1]).map("{:.1%}".format).to_string(), file=sys.stderr)
    return gt, preds


# =============================================================================== execução
@dataclass
class LiveRun:
    out_dir: Path
    run_dir: Path
    dataset: pd.DataFrame
    reference: str
    production_notes: pd.DataFrame
    reference_cost_usd: float


def run_live(cfg: BenchmarkConfig, calls: pd.DataFrame, reference: str, *, label: str = "teste_ligacoes"
             ) -> LiveRun:
    from kpibench.pipeline import run

    if reference in cfg.models:
        print(f"[aviso] {reference} é a referência: removido dos candidatos (concordaria consigo mesmo)",
              file=sys.stderr)
        cfg.models = [m for m in cfg.models if m != reference]
    if not cfg.models:
        raise ValueError("nenhum modelo candidato além da referência")
    calls = calls[calls["resumo"].fillna("").str.len() > 0].copy()
    calls["call_id"] = calls["call_id"].astype(str)

    out_dir = cfg.resolve(cfg.run.output_dir) / f"{label}_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)

    gt, ref_preds = reference_labels(cfg, calls, reference, out_dir)
    prod, notes = map_production_baseline(calls)
    keep = ["call_id", "resumo"] + [c for c in ("transcricao", "data_referencia") if c in calls]
    dataset = calls[keep].merge(gt, on="call_id", how="left").merge(prod, on="call_id", how="left")
    dataset.to_csv(out_dir / "dataset.csv", index=False)

    c = cfg.model_copy(deep=True)
    c.data.path = str(out_dir / "dataset.csv")
    c.data.reference_summary_column = None
    c.data.baseline_prefix = "prod_"
    c.data.sample = None
    if "transcricao" not in dataset.columns or dataset["transcricao"].fillna("").str.len().sum() == 0:
        c.data.transcript_column = None
    c.run.output_dir = str(out_dir)
    c.run.name = "benchmark"
    run_dir = run(c)
    ref_cost = float(ref_preds["cost_usd"].sum()) if not ref_preds.empty else 0.0
    meta = {"referencia": reference, "modelos": c.models, "kpis": [k.name for k in c.selected_kpis],
            "n_ligacoes": int(len(dataset)), "custo_referencia_usd": ref_cost,
            "simulado": bool(c.run.mock), "criado_em": datetime.now().isoformat(timespec="seconds")}
    (out_dir / "teste_ligacoes.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    notes.to_csv(out_dir / "baseline_producao_regras.csv", index=False)
    return LiveRun(out_dir, run_dir, dataset, reference, notes, ref_cost)


# =============================================================================== resultados
@dataclass
class Results:
    metrics: pd.DataFrame      # por KPI × modelo (métrica primária, IC, decisão, custo)
    costs: pd.DataFrame        # por KPI × modelo (tokens, custo, latência)
    routing: pd.DataFrame      # recomendação por KPI
    preds: pd.DataFrame        # predição por ligação × KPI × modelo
    judge: pd.DataFrame
    run_dir: Path


def load_results(run_dir: Path) -> Results:
    def rd(name, **kw):
        p = run_dir / name
        return pd.read_csv(p, **kw) if p.exists() and p.stat().st_size > 1 else pd.DataFrame()
    return Results(rd("metricas_por_kpi_modelo.csv"), rd("custos_latencia.csv"), rd("recomendacao_roteamento.csv"),
                   rd("predictions.csv", dtype={"call_id": str}), rd("judge_scores.csv", dtype={"call_id": str}),
                   run_dir)


def model_scoreboard(res: Results, cfg: BenchmarkConfig, weights: dict[str, float] | None = None) -> pd.DataFrame:
    """Uma linha por modelo candidato: índice de concordância, KPIs não-inferiores, custo, latência, confiabilidade."""
    m = res.metrics[res.metrics["model"] != BASELINE_NAME].copy()
    m["w"] = m["kpi"].map(lambda k: (weights or {}).get(k, 1.0))
    m["score"] = m["primary"].clip(lower=0, upper=1) * 100
    idx = m.groupby("model").apply(lambda g: np.average(g["score"], weights=g["w"]), include_groups=False)
    ok = m.assign(ok=m["decision"].isin(["referencia", "nao_inferior"])).groupby("model")["ok"].sum()
    cost = res.costs.groupby("model")["measured_usd_per_1k"].sum()
    p = res.preds
    lat = p.groupby("model")["latency_s"].agg(lat_p50="median", lat_p95=lambda s: s.quantile(0.95))
    valid = p.groupby("model")["valid"].mean()
    board = pd.DataFrame({"indice": idx, "kpis_nao_inferiores": ok, "usd_1k_ligacoes": cost,
                          "taxa_resposta_valida": valid}).join(lat)
    board["n_kpis"] = m.groupby("model")["kpi"].nunique()
    board = board.reset_index().rename(columns={"index": "model"})
    board["provedor"] = board["model"].map(provider_of)
    board["preco_estimado"] = board["model"].map(lambda x: cfg.model(x).pricing_is_estimate)
    board["indice_por_dolar"] = board["indice"] / board["usd_1k_ligacoes"].replace(0, np.nan)
    return board.sort_values("indice", ascending=False).reset_index(drop=True)


def production_comparison(res: Results) -> pd.DataFrame:
    """KPIs comparáveis: produção atual × melhor modelo × modelo recomendado (mesma referência)."""
    m = res.metrics
    rows = []
    for kpi, g in m.groupby("kpi"):
        prod = g[g["model"] == BASELINE_NAME]
        if prod.empty:
            continue
        cand = g[g["model"] != BASELINE_NAME].sort_values("primary", ascending=False)
        rec = res.routing[res.routing["kpi"] == kpi]
        rec_model = rec["recomendado"].iloc[0] if len(rec) else cand["model"].iloc[0]
        rec_row = cand[cand["model"] == rec_model].iloc[0]
        rows.append({"kpi": kpi, "display": rec["display"].iloc[0] if len(rec) else kpi,
                     "metrica": g["primary_metric"].iloc[0], "producao": float(prod["primary"].iloc[0]),
                     "melhor_modelo": cand["model"].iloc[0], "melhor": float(cand["primary"].iloc[0]),
                     "recomendado": rec_model, "perf_recomendado": float(rec_row["primary"])})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["ganho_recomendado_pp"] = (out["perf_recomendado"] - out["producao"]) * 100
    return out


def executive_summary(board: pd.DataFrame, res: Results, live: LiveRun | None, n_calls: int,
                      period: str) -> str:
    if board.empty:
        return "_Sem resultados._"
    best = board.iloc[0]
    eligible = board[board["kpis_nao_inferiores"] == board["n_kpis"]]
    cheapest_ok = eligible.sort_values("usd_1k_ligacoes").iloc[0] if len(eligible) else None
    value = board.sort_values("indice_por_dolar", ascending=False).iloc[0]
    routed_cost = float(res.routing["usd_1k_recomendado"].sum()) if "usd_1k_recomendado" in res.routing else np.nan
    inconc = int(res.routing["n_inconclusivos"].sum()) if "n_inconclusivos" in res.routing else 0
    from kpibench.reporting.aa_charts import br, usd

    ref = live.reference if live else "referência"
    L = [f"**{n_calls} ligações · {period} · referência: `{ref}`** (métricas = concordância com a referência).", ""]
    L.append(f"- **Maior concordância:** `{best['model']}`, índice {br(best['indice'], 1)}/100, "
             f"{usd(best['usd_1k_ligacoes'])} por 1.000 ligações (todos os KPIs selecionados).")
    if cheapest_ok is not None:
        econ = 100 * (1 - cheapest_ok["usd_1k_ligacoes"] / best["usd_1k_ligacoes"]) if best["usd_1k_ligacoes"] else 0
        L.append(f"- **Mais barato sem perda estatística em nenhum KPI:** `{cheapest_ok['model']}`, "
                 f"{usd(cheapest_ok['usd_1k_ligacoes'])}/1.000 ligações ({br(econ, 0)}% abaixo do líder).")
    else:
        L.append("- **Nenhum modelo é não-inferior ao melhor em todos os KPIs** com esta amostra: "
                 "o roteamento por KPI é o caminho.")
    L.append(f"- **Melhor índice por dólar:** `{value['model']}` ({br(value['indice'], 1)} pontos a "
             f"{usd(value['usd_1k_ligacoes'])}/1.000 ligações).")
    if np.isfinite(routed_cost):
        L.append(f"- **Roteamento recomendado por KPI:** {usd(routed_cost)} por 1.000 ligações "
                 f"(cada KPI no modelo mais barato não-inferior; ver tabela 7.7).")
    slow = board.sort_values("lat_p95").iloc[-1]
    L.append(f"- **Latência:** p95 de {br(board['lat_p95'].min(), 1)} s a {br(slow['lat_p95'], 1)} s por chamada "
             f"de KPI (mais lento: `{slow['model']}`).")
    if board["preco_estimado"].any():
        L.append(f"- **Custo estimado** (sem tabela oficial) para: "
                 f"{', '.join('`'+m+'`' for m in board.loc[board['preco_estimado'], 'model'])}.")
    if inconc:
        L.append(f"- **{inconc} comparações inconclusivas:** a amostra não separa modelos próximos; aumente "
                 f"`NUM_CALLS` antes de decidir.")
    return "\n".join(L)


def call_drilldown(res: Results, dataset: pd.DataFrame, call_id: str, models: list[str] | None = None
                   ) -> pd.DataFrame:
    """KPI × (referência, produção, modelos) para uma ligação."""
    from kpibench.config import load_config as _lc

    kpi_reg = _lc(PROJECT_ROOT / "config" / "benchmark.yaml").kpi_registry
    row = dataset[dataset["call_id"].astype(str) == str(call_id)]
    if row.empty:
        raise KeyError(call_id)
    row = row.iloc[0]
    p = res.preds[res.preds["call_id"] == str(call_id)]
    models = models or sorted(p["model"].unique())
    def norm(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return np.nan
        if isinstance(v, (float, np.floating)) and float(v).is_integer():
            return int(v)
        if isinstance(v, str) and v.replace(".", "", 1).isdigit() and float(v).is_integer():
            return int(float(v))
        return v

    out, text_rows = [], []
    for kname in p["kpi"].unique():
        k = kpi_reg[kname]
        if k.type == "text":
            text_rows.append(k.display)
        r = {"KPI": k.display, "Referência": norm(row.get(k.gt_column)), "Produção": norm(row.get(f"prod_{kname}", np.nan))}
        for m in models:
            v = p[(p["kpi"] == kname) & (p["model"] == m)]
            r[m] = (norm(v["value"].iloc[0]) if bool(v["valid"].iloc[0]) else "⟂ inválido") if len(v) else np.nan
        out.append(r)
    df = pd.DataFrame(out).set_index("KPI")
    df.attrs["text_rows"] = text_rows
    return df



def most_divergent_calls(res: Results, dataset: pd.DataFrame, n: int = 3) -> list[str]:
    """Ligações em que mais modelos discordam da referência (nos KPIs categóricos/ordinais)."""
    from kpibench.config import load_config as _lc

    kpi_reg = _lc(PROJECT_ROOT / "config" / "benchmark.yaml").kpi_registry
    p = res.preds[res.preds["model"] != BASELINE_NAME]
    p = p[p["kpi"].map(lambda k: kpi_reg[k].type != "text")]
    gt = dataset.set_index(dataset["call_id"].astype(str))
    def miss(r):
        ref = gt.at[r.call_id, kpi_reg[r.kpi].gt_column] if r.call_id in gt.index else None
        if ref is None or pd.isna(ref):
            return np.nan
        return float(not r.valid or str(r.value) != str(int(ref) if isinstance(ref, float) else ref))
    p = p.assign(erro=[miss(r) for r in p.itertuples()])
    return p.groupby("call_id")["erro"].mean().sort_values(ascending=False).head(n).index.tolist()
