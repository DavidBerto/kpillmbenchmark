# kpibench — custo × performance de LLMs por KPI de speech analytics

Benchmark para escolher **qual LLM extrair cada KPI** da plataforma de speech analytics
(KPIs extraídos dos resumos das transcrições), medindo concordância com rótulos já
validados, custo projetado em produção e latência — com decisão estatística por KPI.

KPIs: qualidade do resumo · CSAT · razão do CSAT · solicitação resolvida · persuasão/retenção ·
solver reasoning · razão da ligação · first call resolution · empatia · clareza das informações · ODC.

## Início rápido

```bash
pip install -e ".[dev]"                 # núcleo (demo offline)
python -m kpibench demo                 # gera dados sintéticos e roda tudo com modelos simulados
pytest -q

pip install -e ".[llm]"                 # provedores reais
cp .env.example .env                    # preencha base_url e bearer_token (gateway Nexus)
python -m kpibench estimate             # custo do benchmark e de produção, sem chamar API
python -m kpibench run                  # rodada real (config/benchmark.yaml)
python -m kpibench evaluate --run-dir outputs/<rodada>   # reavalia sem chamar API
```

Todos os modelos em `config/models.yaml` usam o gateway **Nexus** (1 endpoint OpenAI-compatível
+ 1 token, expondo múltiplos provedores). Estão registrados os 22 modelos de chat da lista de
`/models` do gateway (OpenAI, Anthropic, DeepSeek, xAI, Moonshot); `text-embedding-3-small` e
`gpt-realtime` ficam de fora por não servirem à extração em JSON. Modelos sem tabela oficial de
preço têm valor **estimado**, comentado no próprio YAML (`# ESTIMATIVA`) e marcado com
`pricing_is_estimate: true` — relatório e notebook sinalizam esses custos como aproximados. Configure `base_url` e `bearer_token` no `.env` — o
`kpibench` carrega o `.env` automaticamente (via `python-dotenv`). As demais variáveis
(`AZURE_OPENAI_*`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`) só são necessárias
se você reativar algum modelo com `provider: azure_openai | openai | anthropic | openai_compatible`.

Saída de cada rodada em `outputs/<nome>_<timestamp>/`: `report.md`, figuras e CSVs.

## Notebooks

Em `notebooks/`:

- `01_validacao_extracao_azure_blob.ipynb` — valida o carregador
  `kpibench.data.azure_blob_loader` (requer `pip install -e ".[azure]"` e
  `az login`): checa se `call_id`/`data_referencia` foram extraídos corretamente
  e se o agrupamento por dia bate com o volume esperado.
- `02_analise_resultados_benchmark.ipynb` — carrega os CSVs da rodada mais
  recente em `outputs/` e monta gráficos de performance por KPI/modelo, custo
  × performance (Pareto) e notas do juiz, além de renderizar o `report.md`.
- `03_teste_ligacoes_nexus.ipynb` — **teste de modelos em ligações reais**. Parâmetros no topo
  (`START_DATE`, `END_DATE`, `NUM_CALLS`, `CALL_IDS`, `MODELOS`, `REFERENCIA`, `PESOS_KPI`,
  `SLA_LATENCIA_S`). Carrega as ligações pela conexão padrão do Azure (`DefaultAzureCredential`,
  via `azure_blob_loader`), estima o custo antes de executar, roda referência e candidatos via
  Nexus e mostra painéis executivos em matplotlib (sem plotly): resumo executivo, ranking, concordância
  × custo com quadrante mais atrativo, custo por 1.000 ligações, latência com SLA, confiabilidade,
  heatmap e IC por KPI, produção atual × recomendado, roteamento por KPI e inspeção lado a lado das
  ligações mais divergentes. `MODO_SIMULADO = True` testa o fluxo sem Azure/Nexus. Fonte do notebook:
  `scripts/build_notebook_03.py`; lógica em `kpibench/live.py`, `kpibench/data/azure_calls.py` e
  `kpibench/reporting/aa_charts.py`.

- `04_extracao_consolidada_transcricao.ipynb` — **extração consolidada**: uma única chamada de LLM
  por ligação (padrão `gpt-6-luna`), sobre a **transcrição completa**, devolvendo **todos os KPIs num
  JSON**. Compara com o fluxo atual (1 chamada por KPI sobre o resumo) nas mesmas ligações e contra a
  mesma referência. Mostra:
  - redução de custo (medido e projetado por mês);
  - consumo de tokens por componente (instruções, conteúdo, saída, raciocínio);
  - sensibilidade ao tamanho da transcrição e latência;
  - qualidade por KPI, com diferença pareada e IC 95%;
  - divergências entre os fluxos.

  Fonte do notebook: `scripts/build_notebook_04.py`. A lógica está em `kpibench/single_call.py` e os gráficos em `kpibench/reporting/single_call_charts.py`. Detalhes na seção "Extração consolidada".

  Sem rótulos auditados na base, a referência padrão é um **modelo forte via Nexus**
  (`REFERENCIA = "claude-opus-5-5"`): as métricas medem concordância com ele, não acerto. Com rótulos
  auditados, use `REFERENCIA = "arquivo:<csv com call_id + gt_*>"`. A produção atual entra como
  baseline só nos KPIs de domínio comparável (CSAT 0–10 convertido para 1–5; resolvida e FCR
  Sim/Não; razão do CSAT e solver reasoning via juiz).

## Desenho

```
arquivo tabular ──► loader (valida/normaliza gt_*)
                      │
   prompts/kpis/<kpi>.md  ── 1 template por KPI, igual para todos os modelos (hash auditado)
                      │
   runner: modelos × KPIs × chamadas  (concorrente, retry c/ backoff, cache SQLite)
                      │
   ├─ parser: envelope tolerante, domínio estrito (inválido = erro, não descartado)
   ├─ juiz LLM p/ KPIs de texto (razão do CSAT, solver reasoning)
   └─ juiz de fidelidade por afirmações (qualidade do resumo)
                      │
   avaliação: métrica primária + IC bootstrap pareado + secundárias
                      │
   custo: medido (usage da API) e projetado (volume/mês, Batch, prompt caching)
                      │
   roteamento: modelo mais barato NÃO-INFERIOR ao melhor, por KPI
                      │
   custo-efetividade: ICER na fronteira de eficiência (qualidade normalizada × custo efetivo) ──► report.md
```

### Métricas por tipo de KPI

| tipo | KPIs | primária | secundárias |
|---|---|---|---|
| ordinal 1–5 | qualidade do resumo, CSAT, empatia, clareza | **QWK** (kappa quadrático) | Spearman, MAE, acerto exato, acerto ±1 |
| categórico | resolvida, razão da ligação, FCR, ODC | **macro-F1** | acurácia, kappa de Cohen, F1 por classe |
| texto | razão do CSAT, solver reasoning | **nota do LLM-juiz** (0–1) vs. referência | chrF++, ROUGE-L, cosseno TF-IDF |

Por que essas: QWK pune erro de 2 pontos 4× mais que de 1 e corrige pelo acaso (acurácia
exata trata 4→5 igual a 1→5); macro-F1 não deixa a classe majoritária mascarar
`parcial`/`nao`; para texto, n-gramas não capturam paráfrase, então o juiz é primário e
as lexicais ficam como sanity check.

### Decisão estatística (roteamento)

Bootstrap pareado (mesmos índices para todos os modelos) da diferença
`melhor − candidato` na métrica primária, com margem de não-inferioridade por KPI
(`ni_margin` em `config/kpis.yaml`):

- **não-inferior**: IC 95% da diferença inteiro abaixo da margem → elegível;
- **inferior**: IC inteiro acima da margem;
- **inconclusivo**: cruza a margem → falta amostra.

Recomendado = elegível mais barato. O relatório também lista o **candidato econômico**
(não inferior pelo ponto, ainda inconclusivo) como hipótese de economia a validar.

### Qualidade do resumo: modernas × eficientes

Cada métrica é correlacionada com a nota humana (Spearman com IC, Kendall τ-b e
AUC para detectar resumos ruins, nota ≤ 2):

| grupo | métrica | precisa de | custo |
|---|---|---|---|
| eficiente | ROUGE-1/2/L, BLEU, chrF++, TF-IDF | referência | CPU, ~US$0 |
| eficiente | precisão ROUGE-2 contra a fonte, taxa de bigramas novos | transcrição | CPU, ~US$0 |
| eficiente | **consistência numérica** (valores, datas, protocolos do resumo presentes na transcrição) | transcrição | CPU, ~US$0 |
| moderna | BERTScore, cosseno de embeddings (multilingual-e5) | referência | GPU/CPU local |
| moderna | SummaC-ZS (NLI multilíngue) | transcrição | GPU/CPU local |
| moderna | **G-Eval** (rubrica CoT + formulário) — é o próprio prompt do KPI, um por modelo | transcrição | API |
| moderna | fidelidade por afirmações atômicas (estilo FActScore/QAG) | transcrição | API |

As métricas neurais exigem `pip install -e ".[neural]"` e acesso ao Hugging Face Hub;
sem isso são marcadas como puladas e o resto segue.

### Custo

- **Medido**: `usage` real de cada chamada × preço de `config/models.yaml`
  (cache read/write e tokens de raciocínio incluídos).
- **Projetado**: média de tokens × volume mensal, com Batch API (KPIs pós-chamada
  toleram latência: −50%) e prompt caching **só** se o system prompt passar do mínimo
  cacheável. Os prompts atuais têm ~400 tokens de prefixo: caching não ajuda aqui.
- Preços têm fonte e data por modelo. **Confira no seu contrato antes de orçar.**

### Custo-efetividade: ICER na fronteira de eficiência

O roteamento responde "qual o modelo mais barato que é **bom o suficiente**". O ICER responde a outra
pergunta: **"quanto custa cada ponto de qualidade a mais?"** Ele substitui a razão ingênua
`primária / custo` (`perf_por_usd`, removida), que:

- explode quando o custo tende a 0, e aí um modelo ruim e quase grátis "vence";
- depende do zero arbitrário de cada métrica (o QWK pode ser negativo; macro-F1 = 0 não é o acaso);
- ignora respostas inválidas e a incerteza da amostra.

Como é calculado (`src/kpibench/analysis/cost_effectiveness.py`):

1. **Qualidade normalizada** `q̃ = (primária − piso) / (teto − piso)`, recortada em [0, 1].
   - O piso padrão é o **acaso**: 0 para QWK e para a nota do juiz; 1/K para macro-F1, que é o F1 esperado de um preditor aleatório que segue a distribuição das K classes presentes.
   - O teto padrão é 1,0. Use a concordância inter-anotador quando ela for medida.
2. **Custo efetivo** `C_ef = custo projetado por 1k chamadas ÷ taxa de respostas válidas`, ou seja, o custo por 1.000 respostas aproveitáveis.
3. **Âncora "acaso"** (custo 0, q̃ 0): um modelo que não supera o acaso é dominado por ela.
4. **Dominância**:
   - **forte**: o modelo custa mais e entrega q̃ menor ou igual a de outro mais barato;
   - **estendida**: o ICER não cresce ao longo da fronteira, então o ponto do meio é superado por uma combinação dos vizinhos.
5. **ICER** = ΔC_ef / Δq̃ entre vizinhos da fronteira, em **US$ por 1.000 chamadas por ponto percentual de q̃**. Compare o ICER com o valor que o negócio atribui a 1 pp de qualidade.
6. **`p_fronteira`**: fração das réplicas bootstrap pareadas (as mesmas do teste de não-inferioridade) em que o modelo fica na fronteira. Mede a robustez da conclusão. O IC do ICER não é reportado porque é instável quando Δq̃ → 0.

O cálculo é feito por KPI e também **agregado por modelo**: custo somado em todos os KPIs (= US$ por 1.000 ligações) e q̃ média ponderada por `cost_effectiveness.kpi_weights`.

Configuração em `benchmark.yaml`:
- `cost_effectiveness.quality_floor` (`chance` ou um número);
- `quality_ceiling`;
- `kpi_weights`.

Por KPI, `quality_floor` e `quality_ceiling` podem ser sobrescritos em `kpis.yaml`.

Saídas:
- `custo_efetividade_por_kpi.csv` e `custo_efetividade_agregado.csv`;
- as colunas `q_norm`, `usd_1k_efetivo`, `status_ce`, `icer_usd_1k_por_pp`, `vizinho_anterior` e `p_fronteira` em `metricas_por_kpi_modelo.csv`;
- a figura `icer_fronteira.png`;
- a seção 6 do `report.md`.

No notebook 03:
- placar (`model_scoreboard`) com as colunas do agregado;
- o gráfico `aa_charts.cost_effectiveness_frontier`;
- o cartão "Fronteira custo-efetiva".

#### Propostas de métricas qualidade × custo avaliadas

| # | Métrica | Fórmula | Prós | Contras | Status |
|---|---|---|---|---|---|
| 1 | Razão simples | Q / C | trivial | explode com C→0; zero arbitrário | removida (`indice_por_dolar` fica só como legado no notebook 03) |
| 2 | Qualidade normalizada | (Q − piso)/(teto − piso) | escalas comparáveis entre KPIs | exige piso e teto | **implementada** (base do ICER) |
| 3 | Custo efetivo por resposta útil | C / taxa válida | inclui o retrabalho | não mede se o conteúdo está correto | **implementada** |
| 4 | Ganho por dólar acima do piso | q̃ / C_ef | interpretável | ainda é uma razão (sensível a C→0) | não implementada |
| 5 | **ICER na fronteira** | ΔC_ef / Δq̃ com dominância forte e estendida | responde "quanto custa cada ponto a mais" | compara só a fronteira | **implementada (principal)** |
| 6 | Benefício líquido (NMB) | λ·q̃ − C_ef | linear, soma entre KPIs | exige λ (disposição a pagar) | extensão futura |
| 7 | CEAC | P_bootstrap(maior NMB \| λ), varrendo λ | incerteza sem fixar λ | leitura menos imediata | extensão futura |
| 8 | Utilidade log-custo | q̃ − α·log10(C) | preços variam em ordens de grandeza | α arbitrário | extensão futura |
| 9 | Versões conservadoras (LCB) | limite inferior do IC de q̃ | penaliza amostra pequena | conservadora | extensão futura |
| 10 | Custo com SLA de latência | C_ef + penalidade se p95 > SLA | inclui a restrição operacional | depende do SLA | extensão futura |

Para implementar NMB e CEAC, reaproveite `normalize_quality`, `effective_cost` e as réplicas de `KPIEvaluation.dists`: o laço de réplicas de `cost_effectiveness()` já tem o formato necessário.

### Extração consolidada (1 chamada por ligação, transcrição completa)

`src/kpibench/single_call.py` mede quanto se economiza, e o que acontece com a qualidade, ao trocar o fluxo
atual pelo consolidado.

| | Fluxo atual (`por_kpi_resumo`) | Fluxo novo (`consolidado_transcricao`) |
|---|---|---|
| Chamadas por ligação | 1 por KPI | 1 |
| Entrada | resumo | transcrição completa (+ resumo só se `qualidade_resumo` estiver incluído) |
| Saída | 1 JSON por KPI | `{"kpis": {"<kpi>": {"justificativa": ..., "valor": ...}, ...}}` |

- **Prompt** (`prompts/consolidado/todos_kpis.md`): reúne as regras gerais, a rubrica de cada KPI e o formato único.
  - As rubricas são extraídas dos próprios `prompts/kpis/<kpi>.md`, sem o bloco "Formato de saída". Mudar um critério num template por KPI muda os dois fluxos.
  - O hash do prompt (`prompt_hash`) cobre o template consolidado e todos os templates por KPI.
  - O system prompt tem cerca de 2,8 mil tokens, acima do mínimo cacheável, então entra no cache da projeção.
- **Parser** (`parse_all_kpis_output`): tolerante no envelope (cercas, KPIs no topo, valor direto) e estrito no domínio, KPI a KPI. Um KPI inválido não derruba os outros; um JSON truncado vira `json_invalido`.
- **`max_tokens`**: a saída traz todos os KPIs, então o estudo eleva o `max_tokens` do modelo (padrão 4.000) e registra `truncado` quando `finish_reason == "length"`.
- **Referência**:
  - `arquivo:<csv>` (rótulos auditados) é a opção sem viés;
  - modelo em modo `consolidado` (transcrição + prompt consolidado; tende a favorecer o fluxo novo);
  - modelo em modo `por_kpi` (resumo + prompts por KPI; tende a favorecer o atual). Rodar os dois modos limita o viés.
- **Qualidade** (`quality_table`): reaproveita `evaluate_kpi`, com bootstrap pareado da diferença novo − atual e margem `ni_margin`. A decisão é `melhora`, `equivalente`, `nao_inferior`, `piora` ou `inconclusivo`. A qualidade agregada usa a qualidade normalizada do ICER (0% = acaso).
- **Custo e tokens** (`cost_table`, `savings_summary`), por abordagem:
  - chamadas por ligação e tokens (instruções, conteúdo, saída, raciocínio, cache);
  - US$/1k ligações medido e projetado (Batch e cache conforme `projection`) e US$/mês;
  - latência sequencial e paralela; % de ligações com todos os KPIs válidos.
  - `summary_cost_usd_per_call` soma ao fluxo atual o custo de gerar o resumo, de que ele depende.
- **Saídas** (`outputs/extracao_consolidada_<timestamp>/`):
  - `saida_consolidada.jsonl` (um JSON por ligação, valores normalizados e flag `valido`);
  - `comparacao_custo_tokens.csv`, `comparacao_qualidade.csv`, `tokens_custo_por_ligacao.csv`;
  - `predicoes.csv`, `juiz.csv`, `referencia_predicoes.csv`, `dataset.csv`, `estudo.json`.
- **Relatório final** (seção 12 do notebook, salvo em `relatorio_final.md`), gerado por `final_report`:
  1. diferença de custo por ligação, por 1k, por mês e por ano (medido e projetado);
  2. cálculo do custo por componente (`cost_breakdown`: tokens médios × preço por 1M), cuja soma reproduz o custo medido (testado);
  3. relação qualidade × custo (`quality_cost_summary`): qualidade normalizada média com IC bootstrap pareado (`aggregate_quality`), custo por ponto de qualidade e veredito (o novo domina, troca qualidade por economia, troca economia por qualidade ou é dominado);
  4. operação e ressalvas.

  Gráficos: `quality_cost_dashboard` (agregado com IC, por KPI e custo por ponto de qualidade) e `cost_calculation` (custo empilhado por componente). No fluxo novo, o custo por KPI é o **rateio igual** da chamada única.
- **Simulação**: o provider mock atende a tarefa `kpi_all`. `MODO_SIMULADO = True` no notebook 04 valida o fluxo sem Azure nem Nexus, mas os números não servem para decisão.

## Estrutura

```
config/        models.yaml (registro + preços) · kpis.yaml (KPIs, taxonomias, margens)
               benchmark.yaml (rodada real) · demo.yaml (simulação offline)
prompts/       _base_system.md · kpis/<kpi>.md (1 por KPI) · consolidado/todos_kpis.md (1 chamada por ligação) · judges/
data/          SCHEMA.md · input/exemplo_kpis_validacao.csv
src/kpibench/  config · data · prompts · llm (providers, cache, pricing) · extraction
               metrics (classificação, texto, stats, summary/) · analysis · reporting · pipeline · cli · live
scripts/       generate_sample_data.py · build_notebook_03.py / build_notebook_04.py (fontes dos notebooks)
tests/         métricas, parser, preços, prompts, custo-efetividade, extração consolidada, notebook 03 e smoke test
```

## Guia para alterações futuras

Mapa do código, por etapa do pipeline (`pipeline.run` → `pipeline.evaluate`):

| Etapa | Arquivo | Pontos de entrada | Saída |
|---|---|---|---|
| Configuração | `config.py` | `load_config`; modelos Pydantic `BenchmarkConfig`, `ModelSpec`, `KPISpec`, `CostEffectivenessCfg` | `cfg` com `model_registry` e `kpi_registry` |
| Dados | `data/loader.py`, `data/azure_blob_loader.py`, `data/azure_calls.py` | `load_dataset` (normaliza `gt_*`, baseline `prod_*`) | `Dataset` |
| Prompts | `prompts/renderer.py` | `PromptRenderer` (hash por template) | mensagens por KPI |
| LLM | `llm/factory.py`, `llm/providers/*`, `llm/cache.py`, `llm/pricing.py` | provider por `ModelSpec.provider` (padrão `nexus`); cache SQLite; `projected_cost_per_call` | `usage` e custo por chamada |
| Extração | `extraction/runner.py`, `extraction/parser.py` | `run_kpi_extraction`, `run_reasoning_judge`, `run_claim_faithfulness` | `predictions.csv`, `judge_scores.csv` |
| Avaliação | `metrics/kpi_eval.py`, `metrics/stats.py`, `metrics/classification.py`, `metrics/text.py` | `evaluate_kpi` → `KPIEvaluation` (`table`, `dists` bootstrap pareadas, `chance`) | métrica primária, IC |
| Custo | `analysis/cost.py` | `cost_latency_table` | `custos_latencia.csv` |
| Roteamento | `analysis/routing.py` | `routing` (não-inferioridade + Pareto + planos) | `recomendacao_roteamento.csv`, `planos_custo_mensal.csv` |
| Custo-efetividade | `analysis/cost_effectiveness.py` | `cost_effectiveness`, `frontier_icer`, `aggregate_point` | `custo_efetividade_*.csv` |
| Resumo | `metrics/summary/*` | `compute_summary_metrics`, `meta_evaluate` | `meta_avaliacao_metricas_resumo.csv` |
| Relatório | `reporting/report.py`, `reporting/figures.py` | `build_report` (seções 1–7) | `report.md` + PNGs |
| Notebook 03 | `live.py`, `reporting/aa_charts.py`, `scripts/build_notebook_03.py` | `run_live`, `load_results`, `model_scoreboard`, `executive_summary` | painéis executivos |
| Extração consolidada (notebook 04) | `single_call.py`, `reporting/single_call_charts.py`, `prompts/consolidado/todos_kpis.md`, `scripts/build_notebook_04.py` | `run_study`, `run_single_call`, `render_single_call`, `estimate_study_cost`, `executive_summary` | `saida_consolidada.jsonl`, `comparacao_*.csv` |

Convenções:

- **Um modelo novo** vai em `config/models.yaml`, com preço, fonte, data e `pricing_is_estimate`. Nenhum código muda.
- **Um KPI novo** precisa de três peças:
  - a entrada em `config/kpis.yaml` (tipo `ordinal`, `categorical` ou `text`; `gt_column`; `ni_margin`);
  - o template `prompts/kpis/<kpi>.md`, que declara o `output_field`;
  - a coluna `gt_<kpi>` nos dados (ver `data/SCHEMA.md`).

  Para o demo, atualize também `scripts/generate_sample_data.py` e `simulation.kpi_difficulty` em `demo.yaml`.
- **A métrica primária** é definida pelo tipo (`KPISpec.primary_metric`). Se mudar, revise o piso "acaso" em `evaluate_kpi` (`chance`) e a margem `ni_margin`.
- **Bootstrap pareado**: todas as distribuições usam os mesmos índices (`bootstrap_indices` com `run.seed`). Toda análise nova de incerteza deve reaproveitar `KPIEvaluation.dists` em vez de reamostrar.
- **`producao_atual`** (`BASELINE_NAME`) aparece nas tabelas de métrica, mas fica fora de custo, roteamento e custo-efetividade, porque não tem custo.
- **Respostas inválidas** contam como erro (`invalid_policy: pessimistic`). O custo efetivo também as penaliza.
- **Reavaliar sem gastar**: `python -m kpibench evaluate --run-dir outputs/<rodada>` refaz avaliação, custo, roteamento, ICER e relatório a partir dos CSVs salvos. Use isso para testar mudanças de análise.
- **Notebooks 03 e 04**: edite `scripts/build_notebook_0X.py` e rode `python scripts/build_notebook_0X.py`. Não edite o `.ipynb` à mão. O builder do 04 não depende de `nbformat`. Regenerar apaga as saídas executadas do notebook.
- **Um KPI novo entra automaticamente no prompt consolidado**: a rubrica vem do template por KPI. O template precisa manter a linha "Formato de saída" para que o bloco de formato seja cortado.
- **Idioma**: código e identificadores em inglês; colunas de saída, relatórios e comentários de domínio em português.

## Pendências conhecidas (nov/2026)

- `scripts/generate_sample_data.py` grava `dash_advins_odc` como `sim`/`nao`, mas `config/kpis.yaml` define o ODC com 7 rótulos. O macro-F1 do ODC vira NaN no demo e `routing` falha com `IndexError` em `econ_pool`, o que quebra `tests/test_pipeline_demo.py`. Alinhe o gerador à taxonomia (ou o contrário) e trate KPIs sem métrica válida no `routing`.
- `tests/test_live.py` espera ao menos 10 modelos com `# ESTIMATIVA` em `models.yaml` e `gpt-6-luna` com `pricing_is_estimate: true`. O `models.yaml` atual não atende a nenhuma das duas condições.
- Arquivos `*:Zone.Identifier` (metadados do Windows) estão versionados ou soltos no repositório. Agora são ignorados pelo `.gitignore`; os já versionados podem ser removidos com `git rm --cached`.
- `exemplo_kpis_validacao.csv` e `report.md` soltos na raiz parecem cópias antigas de `data/input/` e `outputs/`.
- O teto de qualidade do ICER é 1,0 até que a concordância inter-anotador seja medida (`quality_ceiling`).
- `razao_ligacao` está como `type: text` em `kpis.yaml` (sem `labels`), mas `prompts/kpis/razao_ligacao.md` pede "exatamente um rótulo da taxonomia". A taxonomia sai vazia no prompt, tanto no fluxo por KPI quanto no consolidado. Defina se o KPI é texto livre ou categórico e alinhe prompt e config.

## Adaptar para os seus dados

1. Coloque o arquivo em `data/input/` seguindo `data/SCHEMA.md` e aponte `data.path`.
2. Ajuste as taxonomias de `razao_ligacao` e **`odc`** em `config/kpis.yaml` (o ODC do exemplo é uma premissa).
3. Ajuste `deployment`/preços em `config/models.yaml` e a lista `models:` do benchmark.
4. `python -m kpibench estimate` → `python -m kpibench run`.

Tamanho de amostra: com n≈120 os ICs ficam com ~0,13 de largura e quase tudo é
inconclusivo. Para separar modelos a ±0,05, planeje algumas centenas a ~1.000 chamadas
rotuladas, estratificadas por motivo e desfecho.

## Limitações conhecidas

- Juiz da mesma família do candidato pode favorecê-lo: use juiz de outra família e calibre-o contra nota humana.
- O teto de qualquer modelo é a concordância entre auditores humanos: meça o kappa inter-anotador.
- KPIs extraídos do resumo herdam os erros do resumo.
- Consistência numérica não reconhece números por extenso na transcrição (saída típica de ASR).
- O demo é **simulado**: serve para validar o pipeline, não para escolher modelo.
