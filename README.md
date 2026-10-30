# kpibench — custo × performance de LLMs por KPI de speech analytics

Benchmark para escolher **qual LLM extrair cada KPI** da plataforma de speech analytics
(KPIs extraídos dos resumos das transcrições), medindo concordância com rótulos já
validados, custo projetado em produção e latência — com decisão estatística por KPI.

KPIs: qualidade do resumo · CSAT · razão do CSAT · solicitação resolvida · solver reasoning ·
razão da ligação · first call resolution · empatia · clareza das informações · ODC.

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
+ 1 token, expondo múltiplos provedores). Configure `base_url` e `bearer_token` no `.env` — o
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
   roteamento: modelo mais barato NÃO-INFERIOR ao melhor, por KPI ──► report.md
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

## Estrutura

```
config/        models.yaml (registro + preços) · kpis.yaml (KPIs, taxonomias, margens)
               benchmark.yaml (rodada real) · demo.yaml (simulação offline)
prompts/       _base_system.md · kpis/<kpi>.md (1 por KPI) · judges/ (juízes)
data/          SCHEMA.md · input/exemplo_kpis_validacao.csv
src/kpibench/  config · data · prompts · llm (providers, cache, pricing) · extraction
               metrics (classificação, texto, stats, summary/) · analysis · reporting · pipeline · cli
scripts/       generate_sample_data.py
tests/         métricas, parser, preços, prompts e smoke test end-to-end
```

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
