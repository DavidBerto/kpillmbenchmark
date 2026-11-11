"""Gera notebooks/03_teste_ligacoes_nexus.ipynb (fonte versionável do notebook).

    python scripts/build_notebook_03.py
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "03_teste_ligacoes_nexus.ipynb"

cells: list = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


md(r"""
# Teste de modelos em ligações reais (Nexus)

**Pergunta de negócio:** para cada KPI do speech analytics, qual modelo disponível no Nexus entrega o
resultado mais próximo da referência pelo menor custo e dentro do SLA de latência?

O notebook:

1. carrega as ligações do período na base de KPIs do Azure (conexão padrão `DefaultAzureCredential`,
   via `kpibench.data.azure_blob_loader`), com o resumo de cada ligação;
2. estima o custo do teste **antes** de chamar qualquer modelo;
3. gera a **referência** (modelo mais forte via Nexus, ou um arquivo com rótulos auditados);
4. roda os modelos candidatos via Nexus com o **mesmo prompt por KPI** e avalia com o pipeline do
   `kpibench` (métrica primária por KPI, IC por bootstrap pareado, não-inferioridade);
5. mostra o resultado em painéis executivos e permite inspecionar ligações individuais.

> **Como ler:** sem rótulos auditados, as métricas medem **concordância com a referência**, não acerto.
> A produção atual entra na comparação só nos KPIs em que o domínio é comparável (seção 7.6).

**Pré-requisitos:** `az login` feito; `.env` com `base_url` e `bearer_token` do Nexus;
`pip install -e ".[llm,azure]"`. Respostas ficam em cache (`.cache/llm_cache.sqlite`): reexecutar não gera custo.
""")

code(r"""
import sys
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
from kpibench.reporting import aa_charts as charts

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.max_colwidth", 120)
%matplotlib inline


def mostrar(fig):
    # exibe explicitamente: o pipeline troca o backend do matplotlib para Agg ao gerar o report.md
    if fig is not None:
        display(fig)
        plt.close(fig)
""")

md(r"""
## 1. Parâmetros

Ajuste aqui e rode o notebook de cima para baixo. `NUM_CALLS` controla o tamanho (e o custo) do teste:
com poucas dezenas de ligações quase toda comparação entre modelos próximos sai **inconclusiva**; para
decidir, planeje algumas centenas.
""")

code(r"""
# ---- Janela e volume de ligações -------------------------------------------------------------
START_DATE = "2026-09-01"   # "YYYY-MM-DD"; START_DATE = END_DATE = None -> última janela registrada na base
END_DATE   = "2026-10-28"
NUM_CALLS  = 50             # ligações a testar (mínimo 10, regra do azure_blob_loader)
CALL_IDS   = []             # opcional: restringe a ligações específicas da janela (ex.: casos de reclamação)

# ---- Modelos (todos via gateway Nexus; ids de config/models.yaml) --------------------------------
MODELOS = [
    "gpt-6-luna", "gpt-5-mini", "gpt-4.1", "gpt-4o-mini",
    "claude-haiku-5-5", "claude-sonnet-5-5",
    "deepseek-v4-flash", "grok-4.3", "kimi-k2.6",
]
REFERENCIA = "claude-opus-5-5"   # modelo de referência via Nexus, ou "arquivo:<csv com call_id + gt_*>"
JUIZ = "gpt-4o-mini"             # juiz dos KPIs de texto livre (razão do CSAT, solver reasoning)
KPIS = "all"                     # ou lista, ex.: ["csat", "razao_ligacao", "first_call_resolution"]

# ---- Leitura de negócio ----------------------------------------------------------------------
PESOS_KPI = {}                   # peso no índice geral, ex.: {"csat": 2.0, "odc": 1.5}; ausente = 1
SLA_LATENCIA_S = 5.0             # linha de SLA no gráfico de latência (segundos por chamada de KPI)
CASOS_PARA_INSPECIONAR = 3       # ligações com mais divergência exibidas lado a lado

# ---- Execução --------------------------------------------------------------------------------
EXECUTAR = True                  # False: para depois da estimativa de custo
METRICAS_RESUMO_LLM = False      # True: fidelidade por afirmações no resumo (1 chamada extra por ligação)
N_BOOTSTRAP = 1000
MAX_WORKERS = 8                  # chamadas simultâneas ao Nexus
MODO_SIMULADO = False            # True: sem Azure/Nexus (arquivo de exemplo + modelos simulados) para testar o fluxo
""")

md("## 2. Conexão com o Nexus")

code(r"""
cfg = live.build_config(models=MODELOS, kpis=KPIS, judge_model=JUIZ, simulated=MODO_SIMULADO,
                        n_bootstrap=N_BOOTSTRAP, max_workers=MAX_WORKERS,
                        llm_summary_metrics=METRICAS_RESUMO_LLM)

disponiveis = None if MODO_SIMULADO else live.nexus_models()
modelos_df = live.check_models(cfg, MODELOS + [REFERENCIA] if not REFERENCIA.startswith("arquivo:") else MODELOS,
                               disponiveis)
display(modelos_df.style.format({"preco_entrada_usd_1M": "US$ {:.3f}", "preco_saida_usd_1M": "US$ {:.3f}"})
        .hide(axis="index"))

if disponiveis is not None:
    fora = modelos_df.loc[~modelos_df["disponivel_no_nexus"].astype(bool), "modelo"].tolist()
    if fora:
        print(f"Fora do Nexus agora (removidos do teste): {fora}")
        MODELOS = [m for m in MODELOS if m not in fora]
        cfg.models = [m for m in cfg.models if m not in fora]
if modelos_df["preco_estimado"].any():
    print("Preço ESTIMADO (sem tabela oficial conferida):",
          ", ".join(modelos_df.loc[modelos_df["preco_estimado"], "modelo"]))
""")

md(r"""
## 3. Ligações da base de KPIs (Azure, conexão padrão)

`load_calls` reaproveita o `azure_blob_loader` (mesma conta, container e prefixo) e completa cada
ligação com o **texto do resumo** — que é a entrada dos prompts — e, quando houver, da transcrição.
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

com_resumo = calls["resumo"].fillna("").str.len().ge(20) if len(calls) else pd.Series(dtype=bool)
print(f"Ligações carregadas: {len(calls)} · com resumo utilizável: {int(com_resumo.sum())} · período: {PERIODO}")
FOOTER = (f"Fonte: kpibench · {int(com_resumo.sum())} ligações · {PERIODO} · referência: {REFERENCIA} · "
          f"gateway Nexus{' · SIMULADO' if MODO_SIMULADO else ''}")

mostrar(charts.calls_per_day(calls, FOOTER))
display(calls.loc[:, [c for c in ["call_id", "data_referencia", "origem_resumo", "resumo"] if c in calls]].head(5))
""")

md(r"""
**Se o resumo vier vazio:** a estrutura do JSON de resumo não está documentada no repositório e a
extração usa heurísticas. A célula abaixo mostra o esqueleto de um blob real para ajustar
`SUMMARY_KEYS` em `kpibench/data/azure_calls.py` (ou passar `summary_keys=` para `load_calls`).
""")

code(r"""
if not MODO_SIMULADO and len(calls) and (~com_resumo).any():
    from kpibench.data.azure_calls import inspect_blob_structure
    exemplo = calls.loc[~com_resumo].iloc[0]
    alvo = exemplo.get("summary") if str(exemplo.get("summary", "")).startswith("http") else exemplo["blob_name"]
    print(f"Estrutura de: {alvo}\n")
    inspect_blob_structure(alvo)
else:
    print("Todas as ligações têm resumo utilizável — nada a inspecionar.")
""")

md(r"""
## 4. Custo estimado do teste (antes de executar)

Tokens de entrada medidos nos prompts reais destas ligações; saída assumida por tipo de KPI
(+300 tokens de raciocínio nos modelos com `reasoning_effort`). O custo **medido** aparece depois da execução.
""")

code(r"""
ref_model = None if REFERENCIA.startswith("arquivo:") else REFERENCIA
estimativa = live.estimate_cost(cfg, calls[com_resumo], ref_model)
display(estimativa.style.format({"usd_estimado": charts.usd}).hide(axis="index"))
if not EXECUTAR:
    raise SystemExit("EXECUTAR = False: revise a estimativa e mude o parâmetro para rodar o teste.")
""")

md("## 5. Execução (referência + candidatos via Nexus)")

code(r"""
resultado = live.run_live(cfg, calls[com_resumo], REFERENCIA)
res = live.load_results(resultado.run_dir)
placar = live.model_scoreboard(res, cfg, PESOS_KPI)
N = len(resultado.dataset)
print(f"Resultados em: {resultado.run_dir}")
print(f"Custo da referência: {charts.usd(resultado.reference_cost_usd)}")
""")

md("## 6. Resumo executivo")

code(r"""
display(Markdown(live.executive_summary(placar, res, resultado, N, PERIODO)))
mostrar(charts.scorecards(placar, res.routing, N, FOOTER))
""")

md(r"""
## 7. Painéis

### 7.1 Ranking geral
Média da métrica primária dos KPIs selecionados (com `PESOS_KPI`, se definidos), em escala 0–100.
""")
code("mostrar(charts.leaderboard(placar, FOOTER, REFERENCIA))")

md(r"""
### 7.2 Concordância × custo
O quadrante verde reúne quem está acima da concordância mediana **e** abaixo do custo mediano.
Ponto vazado = preço estimado.
""")
code("mostrar(charts.quality_vs_cost(placar, FOOTER))")

md(r"""
### 7.2b Custo-efetividade (ICER)
Qualidade **normalizada** (0% = acaso, 100% = teto) × custo por 1.000 ligações com resposta válida.
Em azul fica a **fronteira de eficiência**; os demais modelos são dominados (há opção mais barata e melhor).
O rótulo de cada segmento é o **ICER**: quanto custa, em US$ por 1.000 ligações, cada ponto percentual de qualidade a mais
frente ao modelo anterior da fronteira. Compare esse valor com quanto vale, para o negócio, 1 pp de qualidade.
""")
code("mostrar(charts.cost_effectiveness_frontier(placar, FOOTER))")

md("### 7.3 Custo por 1.000 ligações")
code("mostrar(charts.cost_bars(placar, FOOTER))")

md("### 7.4 Latência e confiabilidade")
code("""
mostrar(charts.latency_chart(placar, FOOTER, sla_s=SLA_LATENCIA_S))
mostrar(charts.reliability_bars(placar, FOOTER))
""")

md(r"""
### 7.5 Concordância por KPI
O heatmap dá a visão geral; os painéis seguintes mostram a incerteza (IC 95%). Um modelo só é
recomendado se for **não-inferior** ao melhor — se os ICs se sobrepõem muito, falta amostra.
""")
code("""
mostrar(charts.kpi_heatmap(res.metrics, res.routing, FOOTER))
mostrar(charts.kpi_small_multiples(res.metrics, res.routing, FOOTER))
""")

md(r"""
### 7.6 Produção atual × modelo recomendado
Só nos KPIs em que o dado de produção tem domínio comparável. As regras de conversão aplicadas estão na tabela.
""")
code("""
mostrar(charts.production_dumbbell(live.production_comparison(res), FOOTER))
display(resultado.production_notes.style.hide(axis="index"))
""")

md(r"""
### 7.7 Roteamento recomendado por KPI
Cada KPI no modelo mais barato estatisticamente não-inferior ao melhor (`*` = preço estimado).
""")
code("charts.style_routing(res.routing, placar)")

md("Placar consolidado por modelo:")
code("charts.style_scoreboard(placar)")

md(r"""
## 8. Inspeção de ligações

As ligações em que mais modelos discordaram da referência. Verde = igual à referência; vermelho = diferente;
KPIs de texto livre ficam sem cor (são avaliados pelo juiz, não por igualdade literal).
Para ver uma ligação específica: `charts.style_drilldown(live.call_drilldown(res, resultado.dataset, "<call_id>"))`.
""")
code("""
for cid in live.most_divergent_calls(res, resultado.dataset, CASOS_PARA_INSPECIONAR):
    resumo = resultado.dataset.loc[resultado.dataset["call_id"] == cid, "resumo"].iloc[0]
    display(Markdown(f"**Ligação `{cid}`**"))
    display(charts.style_drilldown(live.call_drilldown(res, resultado.dataset, cid), resumo))
""")

md("## 9. Exportar painéis")
code("""
paineis = {
    "01_resumo": charts.scorecards(placar, res.routing, N, FOOTER),
    "02_ranking": charts.leaderboard(placar, FOOTER, REFERENCIA),
    "03_concordancia_custo": charts.quality_vs_cost(placar, FOOTER),
    "03b_custo_efetividade": charts.cost_effectiveness_frontier(placar, FOOTER),
    "04_custo": charts.cost_bars(placar, FOOTER),
    "05_latencia": charts.latency_chart(placar, FOOTER, sla_s=SLA_LATENCIA_S),
    "06_confiabilidade": charts.reliability_bars(placar, FOOTER),
    "07_heatmap_kpi": charts.kpi_heatmap(res.metrics, res.routing, FOOTER),
    "08_kpis_ic": charts.kpi_small_multiples(res.metrics, res.routing, FOOTER),
    "09_producao": charts.production_dumbbell(live.production_comparison(res), FOOTER),
}
destino = resultado.out_dir / "paineis"
destino.mkdir(exist_ok=True)
for nome, fig in paineis.items():
    if fig is not None:
        fig.savefig(destino / f"{nome}.png")
        plt.close(fig)
placar.to_csv(resultado.out_dir / "placar_modelos.csv", index=False)
print(f"Painéis e placar salvos em: {destino.parent}")
print(f"Relatório completo: {resultado.run_dir / 'report.md'}")
""")

md(r"""
## Limitações

- **Concordância ≠ acerto.** Com referência por modelo, um candidato da mesma família da referência tende a
  concordar mais com ela. Para decisão final, use rótulos auditados (`REFERENCIA = "arquivo:..."`) numa amostra.
- **Preços estimados** (marcados em `config/models.yaml`) não incluem eventual markup do Nexus.
- **Amostra:** comparações "inconclusivas" pedem mais ligações, não uma decisão.
- **Latência** medida do notebook até o Nexus; inclui rede e fila do gateway.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
nbf.write(nb, OUT)
print(f"notebook gerado: {OUT}")
