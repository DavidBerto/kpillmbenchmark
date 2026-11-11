"""Gera notebooks/05_comparacao_llm_cdm_jev.ipynb (fonte versionável do notebook).

    python scripts/build_notebook_05.py

Escreve o .ipynb (nbformat 4) diretamente, sem depender do pacote nbformat.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "05_comparacao_llm_cdm_jev.ipynb"

cells: list = []


def _src(text: str) -> list[str]:
    lines = text.strip("\n").split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]


def md(text: str) -> None:
    cells.append({"cell_type": "markdown", "id": uuid.uuid4().hex[:8], "metadata": {}, "source": _src(text)})


def code(text: str) -> None:
    cells.append({"cell_type": "code", "id": uuid.uuid4().hex[:8], "metadata": {}, "execution_count": None,
                  "outputs": [], "source": _src(text)})


md(r"""
# LLM × CDM (ScaleDown) × Jev: qual compensa mais na extração de KPIs

**Pergunta de negócio:** para extrair os KPIs das ligações em produção, compensa mais continuar com o
sistema atual de **LLM** ou trocar pelos **Constrained Decoding Models (CDMs) da ScaleDown**, pelo **Jev da
TypeSafe** ou por uma combinação com LLM? A comparação considera **latência**, **custo** e **relação qualidade × custo**.

| Backend | O que é | Requisições por ligação | Cobre KPIs de texto? |
|---|---|---|---|
| `llm_por_kpi` | **sistema atual**: LLM, 1 prompt por KPI, sobre o resumo | 1 por KPI | sim |
| `llm_consolidado` | LLM, todos os KPIs num JSON pedido no prompt, sobre a transcrição | 1 | sim |
| `llm_estruturado` | LLM com **structured outputs**: a geração fica **restrita a um JSON Schema estrito** gerado dos KPIs (enum nos categóricos, inteiros na escala nos ordinais) | 1 | sim |
| `cdm` | **ScaleDown CDM**: *Decisions API* com perguntas tipadas (`choice` nos categóricos, `score` nos ordinais) numa requisição, e *Summarize* para cada KPI de texto livre; devolve confiança e o custo (`usage.cost`) | 1 + 1 por KPI de texto (em paralelo) | sim (Summarize) |
| `jev` | **TypeSafe Jev**: modelo de decisão tipada com o mesmo formato de perguntas; devolve probabilidades e confiança | 1 | **não** (não gera texto) |
| `cdm_hibrido` | CDM nos KPIs estruturados + LLM nos KPIs de texto e nos KPIs em que o CDM teve **confiança abaixo do limiar** | 1 a 2 | sim |
| `jev_hibrido` | Jev nos KPIs estruturados + LLM nos KPIs de texto e nos KPIs de baixa confiança | 1 a 2 | sim |

Todos rodam nas **mesmas ligações** contra a **mesma referência**. A qualidade é comparada nos KPIs que **todos**
cobrem (os estruturados), e a cobertura de cada backend é informada à parte. Os backends são os mesmos de
produção (`kpibench/extractors.py`, configurados em `config/extractors.yaml`) e podem ser usados direto pelo
comando `python -m kpibench extract` (seção 11).

**Acesso por API key** (no `.env`, nunca no código):

| Backend | Variáveis |
|---|---|
| LLMs (Nexus), inclusive `llm_estruturado` | `base_url`, `bearer_token` |
| CDM ScaleDown | `SCALEDOWN_API_KEY` (API em `https://api.scaledown.xyz`, header `x-api-key`; chave no painel em scaledown.ai) |
| Jev | `TYPESAFE_API_KEY` (ou, via OpenRouter, `OPENROUTER_API_KEY` com `base_url`/`path` ajustados em `extractors.yaml`) |

Lógica: `kpibench/extractors.py` e `kpibench/backend_comparison.py`. Gráficos:
`kpibench/reporting/backend_charts.py`. Fonte deste notebook: `scripts/build_notebook_05.py`.
""")

code(r"""
import sys
import json
import warnings
from pathlib import Path

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Markdown, display

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

from kpibench import live
from kpibench import extractors as X
from kpibench import single_call as sc
from kpibench import backend_comparison as bc
from kpibench.reporting import aa_charts as charts
from kpibench.reporting import backend_charts as bch

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.max_colwidth", 140)
%matplotlib inline


def mostrar(fig):
    if fig is not None:
        display(fig)
        plt.close(fig)
""")

md(r"""
## 1. Parâmetros

Os backends, modelos, endpoints e preços ficam em `config/extractors.yaml`. Aqui você escolhe quais comparar,
a referência e o volume. Com poucas dezenas de ligações, **custo e latência** já ficam estáveis; a **qualidade**
precisa de algumas centenas para separar backends próximos.
""")

code(r"""
# ---- Ligações ----------------------------------------------------------------------------------
START_DATE = "2026-09-01"
END_DATE   = "2026-10-28"
NUM_CALLS  = 50
CALL_IDS   = []

# ---- Backends (nomes de config/extractors.yaml) ------------------------------------------------
BACKENDS = ["llm_por_kpi", "llm_consolidado", "llm_estruturado", "cdm", "cdm_hibrido", "jev", "jev_hibrido"]
BASE     = "llm_por_kpi"          # sistema atual: a referência das diferenças de custo e qualidade

# ---- Referência e juiz -------------------------------------------------------------------------
REFERENCIA      = "claude-opus-5-5"   # modelo forte via Nexus, ou "arquivo:<csv com call_id + colunas gt>"
REFERENCIA_MODO = "consolidado"       # "consolidado" (lê a transcrição) ou "por_kpi" (lê o resumo)
JUIZ            = "gpt-4o-mini"       # juiz dos KPIs de texto livre
KPIS            = "all"
INCLUIR_QUALIDADE_RESUMO = True

# ---- Critérios de decisão ----------------------------------------------------------------------
MARGEM_QUALIDADE_PP = 3.0     # quanto de qualidade (pp, KPIs comuns) aceitamos perder frente à base
SLA_LATENCIA_S      = 3.0     # segundos por ligação para obter todos os KPIs
PESOS_KPI           = {}

# ---- Execução ----------------------------------------------------------------------------------
EXECUTAR      = True
N_BOOTSTRAP   = 1000
MAX_WORKERS   = 8
MODO_SIMULADO = False         # True: arquivo de exemplo + respostas simuladas (testa o fluxo sem APIs)
""")

md("## 2. Configuração dos backends e credenciais")

code(r"""
cfg = live.build_config(models=["gpt-6-luna"], kpis=KPIS, judge_model=JUIZ, simulated=MODO_SIMULADO,
                        n_bootstrap=N_BOOTSTRAP, max_workers=MAX_WORKERS)
conf = X.load_extractor_config(cfg)
display(pd.DataFrame([{"backend": b, "tipo": conf[b]["type"],
                       "modelo/endpoint": conf[b].get("model") or f"{conf[b].get('primary')} + {conf[b].get('secondary')}",
                       "preço entrada US$/1M": (conf[b].get("pricing") or {}).get("input"),
                       "preço saída US$/1M": (conf[b].get("pricing") or {}).get("output"),
                       "preço estimado": conf[b].get("pricing_is_estimate")} for b in BACKENDS]).style.hide(axis="index"))

ctx_check = X.ExtractorContext(cfg, conf=conf)
credenciais = X.credentials_status(ctx_check, BACKENDS)
display(credenciais.style.hide(axis="index"))
if not MODO_SIMULADO and not credenciais["definida"].all():
    faltando = sorted(set(credenciais.loc[~credenciais["definida"], "variavel"]))
    raise RuntimeError(f"defina no .env: {faltando} (ou remova o backend de BACKENDS)")
""")

md("## 3. Ligações")

code(r"""
if MODO_SIMULADO:
    calls = live.sample_calls(NUM_CALLS)
    PERIODO = "arquivo de exemplo (simulado)"
else:
    from kpibench.data.azure_calls import load_calls
    calls = load_calls(START_DATE, END_DATE, NUM_CALLS, call_ids=CALL_IDS or None, max_workers=MAX_WORKERS)
    PERIODO = (f"{pd.to_datetime(calls['data_referencia']).min():%d/%m/%Y} a "
               f"{pd.to_datetime(calls['data_referencia']).max():%d/%m/%Y}") if len(calls) else "—"
for c in ("resumo", "transcricao"):
    calls[c] = calls.get(c, pd.Series("", index=calls.index)).fillna("").astype(str)
print(f"Ligações: {len(calls)} · com transcrição: {int((calls['transcricao'].str.len() > 50).sum())} · "
      f"com resumo: {int((calls['resumo'].str.len() > 20).sum())} · período: {PERIODO}")
FOOTER = (f"Fonte: kpibench · {len(calls)} ligações · {PERIODO} · base: {BASE} · referência: {REFERENCIA}"
          f"{' · SIMULADO' if MODO_SIMULADO else ''}")
""")

md(r"""
## 4. Como cada backend pergunta

- **CDM ScaleDown:** a *Decisions API* recebe a transcrição em `state.text` e uma pergunta tipada por KPI
  estruturado (`choice`/`score`), com a rubrica do KPI em `instructions`. Cada KPI de texto vira uma chamada ao
  *Summarize*, com a rubrica como instrução.
- **Jev:** o mesmo formato de perguntas tipadas (a transcrição vai em `state.transcricao`).
- **LLM com structured outputs:** o mesmo prompt consolidado do LLM, com a geração **restrita** ao JSON Schema
  abaixo, o que torna impossível sair do domínio.
""")

code(r"""
kpis = sc.single_call_kpis(cfg, INCLUIR_QUALIDADE_RESUMO)
exemplo = calls.iloc[0].to_dict()

cdm = X.build_extractor("cdm", ctx_check, kpis)
estruturados = [k for k in kpis if k.type != "text"]
pl = cdm.payload(exemplo, estruturados)
print(f"CDM ScaleDown — Decisions API ({cdm.spec.base_url}{cdm.spec.params['path']}, modelo {cdm.spec.model}): "
      f"{len(pl['questions'])} perguntas ({', '.join(f'{k}={q['type']}' for k, q in pl['questions'].items())})")
print(f"CDM ScaleDown — Summarize ({cdm.summ_spec.model}) para: {[k.name for k in kpis if k.type == 'text']}")
texto = next((k for k in kpis if k.type == "text"), None)
if texto is not None:
    print("\nInstrução do Summarize (" + texto.name + "):\n" + cdm.summarize_prompt(exemplo, texto).system[:700])

schema = X.kpi_json_schema(kpis)
print("\nJSON Schema estrito do LLM com structured outputs (trecho):")
print(json.dumps(schema, ensure_ascii=False, indent=1)[:1200], "...")

jev = X.build_extractor("jev", ctx_check, kpis)
payload = jev.payload(exemplo, jev.supported_kpis)
print(f"\nJev: {len(payload['questions'])} perguntas tipadas "
      f"({', '.join(f'{k}={q['type']}' for k, q in payload['questions'].items())})")
print(f"KPIs que o Jev NÃO cobre (texto livre): {[k.name for k in kpis if not jev.supports(k)]}")
q0 = next(iter(payload["questions"].items()))
print(f"\nExemplo de pergunta ({q0[0]}):\n" + json.dumps(q0[1], ensure_ascii=False, indent=1)[:1200])
""")

md("## 5. Custo estimado do teste (antes de executar)")

code(r"""
estimativa = bc.estimate_backends(cfg, calls, BACKENDS, INCLUIR_QUALIDADE_RESUMO)
display(estimativa.style.format({"usd_estimado": charts.usd, "entrada_tokens_por_ligacao": "{:,.0f}",
                                 "saida_tokens_por_ligacao": "{:,.0f}"}).hide(axis="index"))
print(f"Total dos backends: {charts.usd(estimativa['usd_estimado'].sum())} (+ referência e juiz)")
if not EXECUTAR:
    raise SystemExit("EXECUTAR = False: revise a estimativa e mude o parâmetro para rodar o teste.")
""")

md("## 6. Execução")

code(r"""
estudo = bc.run_backend_comparison(cfg, calls, BACKENDS, baseline=BASE, reference=REFERENCIA,
                                   reference_mode=REFERENCIA_MODO, include_summary_quality=INCLUIR_QUALIDADE_RESUMO,
                                   weights=PESOS_KPI, quality_margin_pp=MARGEM_QUALIDADE_PP, sla_s=SLA_LATENCIA_S,
                                   workers=MAX_WORKERS)
print(f"Resultados em: {estudo.out_dir}")
display(bch.style_summary(estudo.summary))
""")

md(r"""
## 7. Relatório: qual compensa mais

**Regra de decisão:** para **substituir** o sistema atual, o backend precisa cobrir todos os KPIs e não ser
inferior à base na qualidade dos KPIs comuns (margem `MARGEM_QUALIDADE_PP`). Entre os elegíveis, vence o de
**menor custo projetado**, com a latência como desempate. O relatório também aponta o melhor em cada dimensão
e a fronteira de custo-efetividade.
""")

code(r"""
relatorio = bc.final_report(estudo)
display(Markdown(relatorio))
""")

md(r"""
## 8. Relação qualidade × custo

**Esquerda:** qualidade nos KPIs comuns (IC 95%) × custo projetado. A fronteira de eficiência é tracejada e os
dominados aparecem esmaecidos. **Abaixo:** custo por ponto de qualidade.
""")

code(r"""
mostrar(bch.quality_cost_frontier(estudo.summary, FOOTER))
mostrar(bch.cost_per_quality(estudo.summary, FOOTER))
""")

md("## 9. Custo, cálculo do custo e consumo de tokens")

code(r"""
mostrar(bch.cost_bars(estudo.summary, cfg.projection.monthly_calls, FOOTER))
mostrar(bch.cost_breakdown_chart(estudo.breakdown, FOOTER))
mostrar(bch.tokens_chart(estudo.summary, FOOTER))
display(estudo.breakdown[estudo.breakdown["tokens_por_ligacao"] > 0].style.format(
    {"tokens_por_ligacao": "{:,.0f}", "preco_usd_por_1M": "US$ {:.3f}", "usd_por_ligacao": "US$ {:.6f}",
     "usd_1k_ligacoes": charts.usd, "pct_do_custo": "{:.1f}%", "requisicoes_por_ligacao": "{:.1f}"}).hide(axis="index"))

conferencia = pd.DataFrame({
    "soma dos componentes (US$/1k)": estudo.breakdown.groupby("backend")["usd_1k_ligacoes"].sum(),
    "custo medido (US$/1k)": estudo.calls.groupby("backend")["cost_usd"].mean() * 1000})
conferencia["diferença"] = conferencia.iloc[:, 0] - conferencia.iloc[:, 1]
display(conferencia.style.format("{:.5f}"))
""")

md("## 10. Latência e qualidade por KPI")

code(r"""
mostrar(bch.latency_chart(estudo.summary, SLA_LATENCIA_S, FOOTER))
mostrar(bch.quality_heatmap(estudo.quality, estudo.backends, FOOTER))
if not estudo.comparisons.empty:
    mostrar(bch.diff_forest(estudo.comparisons, estudo.summary, BASE, FOOTER))
""")

md(r"""
### 10.1 Confiança × acerto (escolha do limiar dos híbridos)
O CDM da ScaleDown e o Jev devolvem uma confiança por resposta. Se ela for informativa, o acerto cai nas faixas
baixas, e é ali que o híbrido deve reextrair no LLM. A curva mostra, para cada limiar, quantas respostas ficam com
o modelo de decisão e o acerto delas. O limiar atual é `confidence_threshold` em `config/extractors.yaml`.
""")

code(r"""
limiar = next((conf[b].get("confidence_threshold") for b in BACKENDS if conf[b]["type"] == "hybrid"), None)
mostrar(bch.calibration_chart(estudo.calibration, limiar, FOOTER))
""")

md(r"""
## 11. Uso em produção

Os mesmos backends rodam em produção pela CLI ou pelo Python. A saída é **um JSON por ligação** com todos os KPIs,
validados no domínio do KPI, com confiança (CDM/Jev), fonte (no híbrido) e custo/latência da ligação.

```bash
python -m kpibench extract --backend cdm --input ligacoes.csv --output kpis_cdm.jsonl
python -m kpibench extract --backend cdm_hibrido --input ligacoes.csv --output kpis.jsonl
python -m kpibench extract --backend jev --input ligacoes.parquet --output kpis_jev.jsonl --kpis csat,odc
```
""")

code(r"""
print("Python:")
print('''from kpibench import extractors as X
from kpibench.config import load_config
cfg = load_config("config/benchmark.yaml")
ctx = X.ExtractorContext(cfg)
extrator = X.build_extractor("cdm", ctx, cfg.selected_kpis)
resultado = extrator.extract({"call_id": "123", "transcricao": "...", "resumo": "..."})
print(resultado.to_json())''')

recomendado = estudo.recommendation["recomendado"]
with open(estudo.jsonl_path(recomendado), encoding="utf-8") as fh:
    print(f"\nSaída de produção do recomendado ({recomendado}), primeira ligação:")
    print(json.dumps(json.loads(fh.readline()), indent=2, ensure_ascii=False)[:2500])
""")

md("## 12. Exportar")

code(r"""
destino = estudo.out_dir / "paineis"
destino.mkdir(exist_ok=True)
paineis = {
    "01_qualidade_custo": bch.quality_cost_frontier(estudo.summary, FOOTER),
    "02_custo_por_ponto": bch.cost_per_quality(estudo.summary, FOOTER),
    "03_custo": bch.cost_bars(estudo.summary, cfg.projection.monthly_calls, FOOTER),
    "04_calculo_custo": bch.cost_breakdown_chart(estudo.breakdown, FOOTER),
    "05_tokens": bch.tokens_chart(estudo.summary, FOOTER),
    "06_latencia": bch.latency_chart(estudo.summary, SLA_LATENCIA_S, FOOTER),
    "07_qualidade_kpi": bch.quality_heatmap(estudo.quality, estudo.backends, FOOTER),
    "08_diferenca_kpi": bch.diff_forest(estudo.comparisons, estudo.summary, BASE, FOOTER) if not estudo.comparisons.empty else None,
    "09_calibracao": bch.calibration_chart(estudo.calibration, limiar, FOOTER),
}
for nome, fig in paineis.items():
    if fig is not None:
        fig.savefig(destino / f"{nome}.png", bbox_inches="tight")
        plt.close(fig)
(estudo.out_dir / "relatorio.md").write_text(relatorio, encoding="utf-8")
print(f"Relatório, painéis, CSVs e saídas JSONL em: {estudo.out_dir}")
for f in sorted(estudo.out_dir.glob("*.*")):
    print(" -", f.name)
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(f"notebook gerado: {OUT}")
