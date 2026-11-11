"""Gera notebooks/04_extracao_consolidada_transcricao.ipynb (fonte versionável do notebook).

    python scripts/build_notebook_04.py

Escreve o .ipynb (nbformat 4) diretamente, sem depender do pacote nbformat.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "04_extracao_consolidada_transcricao.ipynb"

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
# Extração consolidada: 1 chamada por ligação, sobre a transcrição completa

**Pergunta de negócio:** extrair **todos os KPIs numa única chamada de LLM por ligação**, lendo a
**transcrição completa**, reduz o custo da geração de KPIs e melhora (ou ao menos mantém) a qualidade?

| | Fluxo atual | Fluxo novo |
|---|---|---|
| Chamadas por ligação | 1 por KPI (11) | **1** |
| Entrada | resumo da ligação | **transcrição completa** |
| Saída | 1 JSON por KPI | **1 JSON com todos os KPIs** |
| Modelo | `MODELO_POR_KPI` | `MODELO` (padrão **gpt-6-luna**) |

Os dois fluxos usam as **mesmas rubricas** (o prompt consolidado reaproveita os critérios de
`prompts/kpis/<kpi>.md`), rodam nas **mesmas ligações** e são comparados com a **mesma referência**:

- **Qualidade**: métrica primária por KPI (QWK, macro-F1 ou nota do juiz) contra a referência, com
  diferença **pareada** novo − atual e IC 95% bootstrap. A produção atual entra onde o domínio é comparável.
- **Custo e tokens**: `usage` real de cada chamada × preço de `config/models.yaml`: tokens de entrada
  (instruções × conteúdo), saída, raciocínio; custo medido e projetado para o volume mensal, com Batch e
  prompt caching conforme `projection` em `config/benchmark.yaml`.

> **Sobre a referência:** sem rótulos auditados, a referência é um modelo forte via Nexus. No modo
> `"consolidado"` ele lê a transcrição com o prompt consolidado: vê a ligação inteira, mas compartilha o
> prompt do fluxo novo e **tende a favorecê-lo**. No modo `"por_kpi"` ele lê o resumo com os prompts por
> KPI e **tende a favorecer o fluxo atual**. Rodar os dois modos dá um intervalo honesto; para decisão final,
> use rótulos auditados (`REFERENCIA = "arquivo:<csv com call_id + colunas gt>"`).

**Pré-requisitos:** `az login`; `.env` com `base_url` e `bearer_token` do Nexus; `pip install -e ".[llm,azure]"`.
Respostas ficam em cache (`.cache/llm_cache.sqlite`): reexecutar não gera custo. Lógica em
`kpibench/single_call.py` e gráficos em `kpibench/reporting/single_call_charts.py`; fonte deste notebook:
`scripts/build_notebook_04.py`.
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
from kpibench import single_call as sc
from kpibench.llm.tokens import estimate_tokens
from kpibench.prompts.renderer import PromptRenderer
from kpibench.reporting import aa_charts as charts
from kpibench.reporting import single_call_charts as scc

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

`NUM_CALLS` controla o tamanho e o custo do teste. Com poucas dezenas de ligações a comparação de
qualidade sai quase toda **inconclusiva**; a medição de **custo e tokens** já é estável com poucas dezenas.
""")

code(r"""
# ---- Janela e volume de ligações ----------------------------------------------------------------
START_DATE = "2026-09-01"     # "YYYY-MM-DD"; None/None -> última janela registrada na base
END_DATE   = "2026-10-28"
NUM_CALLS  = 50               # ligações a testar (mínimo 10)
CALL_IDS   = []               # opcional: restringe a ligações específicas

# ---- Modelos (ids de config/models.yaml, todos via Nexus) ------------------------------------------
MODELO          = "gpt-6-luna"        # fluxo novo: 1 chamada por ligação sobre a transcrição
MODELO_POR_KPI  = "gpt-6-luna"        # fluxo atual: 1 chamada por KPI sobre o resumo (mesmo modelo isola o efeito do desenho)
REFERENCIA      = "claude-opus-5-5"   # modelo forte via Nexus, ou "arquivo:<csv com call_id + colunas gt>"
REFERENCIA_MODO = "consolidado"       # "consolidado" (lê a transcrição) ou "por_kpi" (lê o resumo)
JUIZ            = "gpt-4o-mini"       # juiz dos KPIs de texto livre
KPIS            = "all"               # ou lista, ex.: ["csat", "solicitacao_resolvida", "odc"]

# ---- Prompt consolidado -----------------------------------------------------------------------
INCLUIR_QUALIDADE_RESUMO = True   # avalia o resumo contra a transcrição na mesma chamada (envia o resumo junto)
JUSTIFICATIVAS           = True   # False: só os valores (menos tokens de saída, menos auditável)
MAX_PALAVRAS             = 25     # limite das justificativas no JSON consolidado
MAX_TOKENS_CONSOLIDADO   = 4000   # saída maior que a de 1 KPI: evita JSON truncado

# ---- Custo ------------------------------------------------------------------------------------
CUSTO_RESUMO_USD_POR_LIGACAO = None   # opcional: custo de gerar o resumo (o fluxo atual depende dele)
PESOS_KPI = {}                        # peso de cada KPI na qualidade agregada; ausente = 1

# ---- Execução ---------------------------------------------------------------------------------
EXECUTAR      = True
N_BOOTSTRAP   = 1000
MAX_WORKERS   = 8
MODO_SIMULADO = False   # True: arquivo de exemplo + modelos simulados (testa o fluxo sem Azure/Nexus)
""")

md("## 2. Configuração e conexão com o Nexus")

code(r"""
cfg = live.build_config(models=[MODELO], kpis=KPIS, judge_model=JUIZ, simulated=MODO_SIMULADO,
                        n_bootstrap=N_BOOTSTRAP, max_workers=MAX_WORKERS)
usados = sorted({MODELO, MODELO_POR_KPI, JUIZ} | ({REFERENCIA} if not REFERENCIA.startswith("arquivo:") else set()))
disponiveis = None if MODO_SIMULADO else live.nexus_models()
modelos_df = live.check_models(cfg, usados, disponiveis)
display(modelos_df.style.format({"preco_entrada_usd_1M": "US$ {:.3f}", "preco_saida_usd_1M": "US$ {:.3f}"})
        .hide(axis="index"))
if disponiveis is not None and not modelos_df["disponivel_no_nexus"].astype(bool).all():
    raise RuntimeError("modelo indisponível no Nexus: ajuste os parâmetros")
spec = cfg.model(MODELO)
print(f"{MODELO}: entrada US$ {spec.pricing.input}/1M · cache US$ {spec.pricing.cached_input}/1M · "
      f"saída US$ {spec.pricing.output}/1M · parâmetros {spec.params} (max_tokens no consolidado: {MAX_TOKENS_CONSOLIDADO})")
print(f"Projeção: {cfg.projection.monthly_calls:,} ligações/mês · Batch: {cfg.projection.use_batch} · "
      f"cache hit: {cfg.projection.cache_hit_ratio:.0%}")
""")

md(r"""
## 3. Ligações (Azure, conexão padrão)

O fluxo novo precisa da **transcrição**; o atual, do **resumo**. Só entram ligações com os dois, para
medir os fluxos nas mesmas ligações.
""")

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
utilizavel = (calls["resumo"].str.len() >= 20) & (calls["transcricao"].str.len() >= 50)
print(f"Ligações: {len(calls)} · com resumo: {int((calls['resumo'].str.len() >= 20).sum())} · "
      f"com transcrição: {int((calls['transcricao'].str.len() >= 50).sum())} · utilizáveis: {int(utilizavel.sum())}")
calls = calls[utilizavel].reset_index(drop=True)
FOOTER = (f"Fonte: kpibench · {len(calls)} ligações · {PERIODO} · {MODELO} · referência: {REFERENCIA} "
          f"({REFERENCIA_MODO}){' · SIMULADO' if MODO_SIMULADO else ''}")

tam = pd.DataFrame({"transcrição (caracteres)": calls["transcricao"].str.len(),
                    "resumo (caracteres)": calls["resumo"].str.len()})
display(tam.describe(percentiles=[.5, .9]).T.style.format("{:,.0f}"))
""")

md(r"""
## 4. O prompt consolidado

Instruções gerais + a rubrica de cada KPI (extraída do template por KPI, sem o bloco de formato) + o
formato do JSON único. A parte fixa (system prompt) é igual para todas as ligações e, por passar do
mínimo cacheável, pode ser servida do **cache de prompt** em produção.
""")

code(r"""
renderer = PromptRenderer(cfg.root / "prompts")
kpis_consolidado = sc.single_call_kpis(cfg, INCLUIR_QUALIDADE_RESUMO)
exemplo = sc.render_single_call(renderer, kpis_consolidado, calls.loc[0, "transcricao"], calls.loc[0, "resumo"],
                                MAX_PALAVRAS, JUSTIFICATIVAS)
print(f"KPIs no JSON: {[k.name for k in kpis_consolidado]}")
print(f"System prompt (fixo): {estimate_tokens(exemplo.system):,} tokens · conteúdo desta ligação: "
      f"{estimate_tokens(exemplo.user):,} tokens · hash do prompt: {exemplo.template_hash}")
print(f"Mínimo cacheável de {MODELO}: {spec.min_cacheable_tokens:,} tokens")
print("\n" + "=" * 100 + "\n" + exemplo.system[-2200:])
""")

md(r"""
## 5. Custo estimado do teste (antes de executar)

Entrada medida nos prompts reais destas ligações; saída assumida (+ raciocínio nos modelos com
`reasoning_effort`). O custo **medido** aparece depois da execução.
""")

code(r"""
estimativa = sc.estimate_study_cost(cfg, calls, model_single=MODELO, model_per_kpi=MODELO_POR_KPI,
                                    reference=REFERENCIA, reference_mode=REFERENCIA_MODO,
                                    include_summary_quality=INCLUIR_QUALIDADE_RESUMO, max_palavras=MAX_PALAVRAS)
display(estimativa.style.format({"usd_estimado": charts.usd, "entrada_tokens_por_ligacao": "{:,.0f}",
                                 "saida_tokens_por_ligacao": "{:,.0f}"}, na_rep="").hide(axis="index"))
if not EXECUTAR:
    raise SystemExit("EXECUTAR = False: revise a estimativa e mude o parâmetro para rodar o teste.")
""")

md("## 6. Execução: referência, fluxo atual e fluxo novo")

code(r"""
estudo = sc.run_study(cfg, calls, model_single=MODELO, model_per_kpi=MODELO_POR_KPI, reference=REFERENCIA,
                      reference_mode=REFERENCIA_MODO, include_summary_quality=INCLUIR_QUALIDADE_RESUMO,
                      max_tokens=MAX_TOKENS_CONSOLIDADO, max_palavras=MAX_PALAVRAS, justificativas=JUSTIFICATIVAS,
                      summary_cost_usd_per_call=CUSTO_RESUMO_USD_POR_LIGACAO, kpi_weights=PESOS_KPI)
print(f"Resultados em: {estudo.out_dir}")
print(f"Custo da referência: {charts.usd(estudo.reference_cost_usd)}")
""")

md("## 7. Resumo executivo")

code(r"""
display(Markdown(sc.executive_summary(estudo)))
mostrar(scc.cost_quality_tradeoff(estudo.cost, estudo.savings, FOOTER))
""")

md(r"""
## 8. A saída: um JSON por ligação

Cada linha de `saida_consolidada.jsonl` é o JSON de uma ligação com todos os KPIs, já validados e
normalizados (`valido = false` quando o valor saiu fora do domínio do KPI).
""")

code(r"""
saidas = [json.loads(l) for l in open(estudo.json_path, encoding="utf-8")]
completas = [s for s in saidas if all(v["valido"] for v in s["kpis"].values())]
print(f"{len(completas)} de {len(saidas)} ligações com todos os KPIs válidos. Exemplo:\n")
print(json.dumps((completas or saidas)[0], indent=2, ensure_ascii=False))

tabela = pd.DataFrame([{"call_id": s["call_id"], **{k: v["valor"] for k, v in s["kpis"].items()}} for s in saidas])
display(tabela.head(10))
""")

md(r"""
## 9. Custo e consumo de tokens

### 9.1 Custo por 1.000 ligações e projeção mensal
**Medido** = `usage` real desta amostra. **Projetado** = médias de tokens × preço, com Batch e cache do
prefixo estático conforme `projection` (o cache só se aplica se o system prompt passar do mínimo cacheável).
""")

code(r"""
mostrar(scc.cost_comparison(estudo.cost, estudo.savings, FOOTER))
mostrar(scc.monthly_projection(estudo.cost, estudo.savings, cfg.projection.monthly_calls, FOOTER))
scc.style_cost(estudo.cost)
""")

md(r"""
### 9.2 Para onde vão os tokens
No fluxo atual, as instruções e o resumo são reenviados a cada KPI; no novo, a transcrição (maior que o
resumo) é lida uma vez e as instruções dos KPIs vão num único system prompt.
""")

code(r"""
mostrar(scc.token_breakdown(estudo.cost, FOOTER))
reducoes = pd.Series({k: v for k, v in estudo.savings.items() if k.startswith("reducao")}, name="redução (%)")
display(reducoes.to_frame().style.format("{:.1f}%"))
""")

md("Fluxo atual: tokens e custo médios por KPI (cada KPI é uma chamada separada).")
code("display(sc.tokens_by_kpi(estudo.preds).style.format(precision=1).hide(axis='index'))")

md(r"""
### 9.3 Sensibilidade ao tamanho da transcrição
O custo do fluxo novo cresce com a transcrição; o do atual é quase fixo (depende do resumo). O
cruzamento das retas indica a partir de que tamanho de ligação o fluxo atual volta a ser mais barato.
""")

code(r"""
mostrar(scc.tokens_vs_transcript(estudo.calls, FOOTER))
novo = estudo.calls[estudo.calls["abordagem"] == sc.SINGLE]
print(f"Transcrições: p50 {novo['transcricao_chars'].median():,.0f} · p90 {novo['transcricao_chars'].quantile(.9):,.0f} "
      f"caracteres · ligações truncadas por max_tokens: {int(novo['truncado'].sum())}")
""")

md("### 9.4 Latência por ligação")
code("mostrar(scc.latency_comparison(estudo.cost, FOOTER))")

md(r"""
## 10. Qualidade

Concordância com a referência por KPI. **Decisão** (IC 95% da diferença novo − atual, bootstrap pareado):
`melhora` (IC acima de 0) · `equivalente` (IC dentro de ±margem do KPI) · `não-inferior` (IC acima de
−margem) · `piora` (IC abaixo de −margem) · `inconclusivo` (falta amostra).
""")

code(r"""
mostrar(scc.quality_forest(estudo.quality, FOOTER))
mostrar(scc.quality_by_kpi(estudo.quality, FOOTER))
scc.style_quality(estudo.quality)
""")

md(r"""
### 10.1 Ligações em que os fluxos divergem
Para inspecionar casos: o valor da referência, do fluxo atual e do novo, lado a lado.
""")

code(r"""
p = estudo.preds.pivot_table(index=["call_id", "kpi"], columns="model", values="value", aggfunc="first")
gt_cols = {k.name: k.gt_column for k in cfg.selected_kpis}
ref = estudo.dataset.set_index("call_id")
p["referencia"] = [ref.at[c, gt_cols[k]] if gt_cols[k] in ref else None for c, k in p.index]
p = p.reset_index()
nao_texto = {k.name for k in cfg.selected_kpis if k.type != "text"}
diverge = p[p["kpi"].isin(nao_texto) & (p[sc.SINGLE].astype(str) != p[sc.PER_KPI].astype(str))]
print(f"{len(diverge)} pares ligação × KPI (não-texto) em que os fluxos discordam")
display(diverge.head(20))
""")

md("## 11. Exportar")

code(r"""
destino = estudo.out_dir / "paineis"
destino.mkdir(exist_ok=True)
paineis = {
    "01_custo_qualidade": scc.cost_quality_tradeoff(estudo.cost, estudo.savings, FOOTER),
    "02_custo_1k": scc.cost_comparison(estudo.cost, estudo.savings, FOOTER),
    "03_custo_mensal": scc.monthly_projection(estudo.cost, estudo.savings, cfg.projection.monthly_calls, FOOTER),
    "04_tokens": scc.token_breakdown(estudo.cost, FOOTER),
    "05_tamanho_transcricao": scc.tokens_vs_transcript(estudo.calls, FOOTER),
    "06_latencia": scc.latency_comparison(estudo.cost, FOOTER),
    "07_qualidade_diferenca": scc.quality_forest(estudo.quality, FOOTER),
    "08_qualidade_kpi": scc.quality_by_kpi(estudo.quality, FOOTER),
}
for nome, fig in paineis.items():
    fig.savefig(destino / f"{nome}.png")
    plt.close(fig)
(estudo.out_dir / "resumo_executivo.md").write_text(sc.executive_summary(estudo), encoding="utf-8")
print(f"Painéis, CSVs e JSONL em: {estudo.out_dir}")
for f in sorted(estudo.out_dir.glob("*.*")):
    print(" -", f.name)
""")

md(r"""
## 12. Relatório final: uma LLM por KPI × todos os KPIs consolidados

Consolida a decisão em um relatório (salvo em `relatorio_final.md` na pasta da rodada):

1. **Diferença de custo**: por ligação, por 1.000 ligações, por mês e por ano (medido e projetado).
2. **Cálculo do custo e consumo de tokens**: tokens médios de cada componente × preço por 1M tokens.
   A soma reproduz o custo medido.
3. **Relação qualidade × custo**: qualidade normalizada média (0% = acaso) com IC 95% bootstrap,
   custo por ponto de qualidade e veredito (o novo domina, troca qualidade por economia ou é dominado).
4. **Operação e ressalvas**: confiabilidade do JSON, latência, tamanho da transcrição e viés da referência.
""")

code(r"""
relatorio = sc.final_report(cfg, estudo, PESOS_KPI)
display(Markdown(relatorio))
""")

md(r"""
### 12.1 Gráfico comparativo: relação qualidade × custo
**Esquerda:** todos os KPIs, com a qualidade média e o IC 95% de cada fluxo; a seta vai do atual ao novo, e o canto
superior esquerdo é o melhor. **Centro:** o mesmo, KPI a KPI (no fluxo novo, o custo da chamada única é
rateado igualmente entre os KPIs). **Direita:** custo por ponto de qualidade (menor é melhor).
""")

code(r"""
resumo_qc = sc.quality_cost_summary(cfg, estudo, PESOS_KPI)
por_kpi_qc = sc.per_kpi_cost_quality(estudo)
mostrar(scc.quality_cost_dashboard(resumo_qc, por_kpi_qc, FOOTER))
display(por_kpi_qc.style.format({"usd_1k_atual": charts.usd, "usd_1k_novo": charts.usd,
                                 "q_atual": "{:.1%}", "q_novo": "{:.1%}"}).hide(axis="index"))
""")

md(r"""
### 12.2 Cálculo do custo e consumo de tokens
Custo por ligação = Σ (tokens médios do componente × preço por 1M tokens) ÷ 1.000.000. A conferência
abaixo compara a soma dos componentes com o custo medido chamada a chamada.
""")

code(r"""
calculo = sc.cost_breakdown(cfg, estudo)
mostrar(scc.cost_calculation(calculo, FOOTER))
display(scc.style_breakdown(calculo))

conferencia = pd.DataFrame({
    "soma dos componentes (US$/1k)": calculo.groupby("abordagem")["usd_1k_ligacoes"].sum(),
    "custo medido (US$/1k)": estudo.calls.groupby("abordagem")["cost_usd"].mean() * 1000,
})
conferencia["diferença"] = conferencia.iloc[:, 0] - conferencia.iloc[:, 1]
display(conferencia.style.format("{:.4f}"))

mostrar(scc.token_breakdown(estudo.cost, FOOTER))
mostrar(scc.cost_comparison(estudo.cost, estudo.savings, FOOTER))
""")

code(r"""
(estudo.out_dir / "relatorio_final.md").write_text(relatorio, encoding="utf-8")
finais = {"12_qualidade_custo": scc.quality_cost_dashboard(resumo_qc, por_kpi_qc, FOOTER),
          "13_calculo_custo": scc.cost_calculation(calculo, FOOTER)}
for nome, fig in finais.items():
    fig.savefig(estudo.out_dir / "paineis" / f"{nome}.png", bbox_inches="tight")
    plt.close(fig)
calculo.to_csv(estudo.out_dir / "calculo_custo_componentes.csv", index=False)
por_kpi_qc.to_csv(estudo.out_dir / "qualidade_custo_por_kpi.csv", index=False)
print(f"Relatório final: {estudo.out_dir / 'relatorio_final.md'}")
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(f"notebook gerado: {OUT}")
