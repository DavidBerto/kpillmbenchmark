"""Linha de comando.

  python -m kpibench demo                                   # gera dados de exemplo e roda tudo offline
  python -m kpibench estimate --config config/benchmark.yaml  # custo estimado, sem chamar API
  python -m kpibench run      --config config/benchmark.yaml  # rodada real
  python -m kpibench evaluate --config config/benchmark.yaml --run-dir outputs/<rodada>
  python -m kpibench extract  --backend jev --input ligacoes.csv --output kpis.jsonl   # produção
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from kpibench.config import PROJECT_ROOT, load_config

OUT_TOKENS_DEFAULT = {"ordinal": 60, "categorical": 60, "text": 70}
OUT_TOKENS_SUMMARY_QUALITY = 250


def cmd_estimate(args) -> int:
    from kpibench.data.loader import load_dataset
    from kpibench.extraction.runner import kpi_prompt_vars
    from kpibench.llm.pricing import projected_cost_per_call
    from kpibench.llm.tokens import estimate_tokens
    from kpibench.prompts.renderer import PromptRenderer

    cfg = load_config(args.config)
    ds = load_dataset(cfg)
    renderer = PromptRenderer(cfg.root / "prompts")
    pj = cfg.projection
    rows = []
    for kpi in cfg.selected_kpis:
        prompts = [renderer.render(f"kpis/{kpi.prompt}", **kpi_prompt_vars(kpi, ds, ds.record(c))) for c in ds.ids()]
        in_tok = sum(estimate_tokens(p.system) + estimate_tokens(p.user) for p in prompts) / len(prompts)
        prefix = prompts[0].static_prefix_tokens
        base_out = OUT_TOKENS_SUMMARY_QUALITY if kpi.name == "qualidade_resumo" else OUT_TOKENS_DEFAULT[kpi.type]
        for spec in cfg.selected_models:
            reasoning = args.reasoning_overhead if "reasoning_effort" in spec.params else 0
            out_tok = base_out + reasoning
            bench = projected_cost_per_call(spec, in_tok, out_tok, 0, 0, batch=False) * len(prompts)
            prod = projected_cost_per_call(spec, in_tok, out_tok, prefix, pj.cache_hit_ratio, batch=pj.use_batch)
            rows.append({"kpi": kpi.name, "model": spec.id, "in_tok": in_tok, "out_tok": out_tok,
                         "prefix_tok": prefix, "usd_benchmark": bench, "usd_mes_producao": prod * pj.monthly_calls})
    df = pd.DataFrame(rows)
    by_model = df.groupby("model")[["usd_benchmark", "usd_mes_producao"]].sum().sort_values("usd_mes_producao")
    print(f"\nEstimativa ({len(ds.df)} registros; tokens estimados; saída assumida por tipo de KPI"
          f" + {args.reasoning_overhead} tokens de raciocínio p/ modelos reasoning)\n")
    print(by_model.to_markdown(floatfmt=",.2f"))
    print(f"\nCusto total estimado do benchmark (sem juízes): US$ {df['usd_benchmark'].sum():,.2f}")
    print(f"Prefixo estático médio: {df['prefix_tok'].mean():.0f} tokens "
          f"({'cacheável' if df['prefix_tok'].mean() >= 1024 else 'abaixo do mínimo de cache'})")
    if args.out:
        df.to_csv(args.out, index=False)
    return 0


def cmd_run(args) -> int:
    from kpibench.pipeline import run

    cfg = load_config(args.config)
    out = run(cfg)
    print(f"Rodada salva em: {out}")
    return 0


def cmd_evaluate(args) -> int:
    from kpibench.pipeline import evaluate_existing

    cfg = load_config(args.config)
    evaluate_existing(cfg, Path(args.run_dir))
    return 0


def cmd_demo(args) -> int:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from generate_sample_data import main as gen  # type: ignore

    data_path = PROJECT_ROOT / "data" / "input" / "exemplo_kpis_validacao.csv"
    if args.regenerate or not data_path.exists():
        gen(n=args.n, out=data_path)
    from kpibench.pipeline import run

    cfg = load_config(PROJECT_ROOT / "config" / "demo.yaml")
    out = run(cfg)
    print(f"Demo salva em: {out}")
    return 0


def cmd_extract(args) -> int:
    """Produção: extrai os KPIs das ligações com um backend de config/extractors.yaml e grava JSONL."""
    from kpibench import extractors as X
    from kpibench.data.loader import _read_any

    cfg = load_config(args.config)
    if args.simulated:
        cfg = load_config(PROJECT_ROOT / "config" / "demo.yaml")
    cfg.run.mock = bool(args.simulated)
    if args.kpis:
        cfg.kpis = [k.strip() for k in args.kpis.split(",") if k.strip()]
    df = _read_any(Path(args.input))
    df["call_id"] = df[args.id_column].astype(str)
    for c in ("transcricao", "resumo"):
        if c not in df:
            df[c] = ""
    ctx = X.ExtractorContext(cfg, conf=X.load_extractor_config(cfg, args.extractors))
    ex = X.build_extractor(args.backend, ctx, cfg.selected_kpis)
    exts = X.run_extractor(ex, df.to_dict("records"), args.workers or cfg.run.max_workers)
    out = X.write_jsonl(exts, Path(args.output))
    per_call, preds, _ = X.extraction_frames(exts)
    lat = per_call["latency_s"].dropna()
    print(f"{len(exts)} ligações -> {out}")
    print(f"backend {args.backend} ({ex.kind}) · KPIs válidos {preds['valid'].mean():.1%} · "
          f"US$ {per_call['cost_usd'].sum():.6f} no lote (US$ {per_call['cost_usd'].mean() * 1000:.4f}/1.000 ligações) · "
          f"latência p50 {lat.median():.2f} s · p95 {lat.quantile(0.95):.2f} s" if len(lat) else "sem latência medida")
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    p = argparse.ArgumentParser(prog="kpibench", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("estimate", help="estima custo do benchmark e de produção sem chamar APIs")
    s.add_argument("--config", default=str(PROJECT_ROOT / "config" / "benchmark.yaml"))
    s.add_argument("--reasoning-overhead", type=int, default=300)
    s.add_argument("--out", default=None, help="CSV opcional com o detalhamento")
    s.set_defaults(func=cmd_estimate)

    s = sub.add_parser("run", help="executa a rodada completa")
    s.add_argument("--config", default=str(PROJECT_ROOT / "config" / "benchmark.yaml"))
    s.set_defaults(func=cmd_run)

    s = sub.add_parser("evaluate", help="reavalia uma rodada salva sem chamar APIs")
    s.add_argument("--config", default=str(PROJECT_ROOT / "config" / "benchmark.yaml"))
    s.add_argument("--run-dir", required=True)
    s.set_defaults(func=cmd_evaluate)

    s = sub.add_parser("demo", help="gera dados sintéticos e roda o pipeline offline (modelos simulados)")
    s.add_argument("--n", type=int, default=120)
    s.add_argument("--regenerate", action="store_true")
    s.set_defaults(func=cmd_demo)

    s = sub.add_parser("extract", help="produção: extrai KPIs com um backend (llm/cdm/jev/híbrido) e grava JSONL")
    s.add_argument("--backend", required=True, help="nome em config/extractors.yaml (ex.: jev, cdm, jev_hibrido)")
    s.add_argument("--input", required=True, help=".csv/.xlsx/.parquet com call_id, transcricao e/ou resumo")
    s.add_argument("--output", required=True, help="arquivo .jsonl de saída (um JSON por ligação)")
    s.add_argument("--config", default=str(PROJECT_ROOT / "config" / "benchmark.yaml"))
    s.add_argument("--extractors", default=None, help="caminho alternativo para extractors.yaml")
    s.add_argument("--kpis", default=None, help="lista separada por vírgula (padrão: todos de kpis.yaml)")
    s.add_argument("--id-column", default="call_id")
    s.add_argument("--workers", type=int, default=None)
    s.add_argument("--simulated", action="store_true", help="provider simulado (sem chamar APIs)")
    s.set_defaults(func=cmd_extract)

    args = p.parse_args(argv)
    return args.func(args)
