# Esquema do arquivo de entrada

Formatos aceitos: `.csv` (UTF-8), `.xlsx`, `.parquet`. Os nomes das colunas de id,
resumo, transcrição e referência são configuráveis em `data:` no YAML do benchmark;
as colunas `gt_*` são definidas em `config/kpis.yaml` (`gt_column`).

| coluna | obrigatória | tipo | descrição |
|---|---|---|---|
| `call_id` | sim | texto único | identificador da chamada |
| `resumo` | sim | texto | resumo gerado pela plataforma — **entrada dos prompts de KPI** |
| `transcricao` | recomendada | texto | transcrição diarizada; habilita fidelidade (G-Eval com fonte, SummaC, claims, consistência numérica) |
| `resumo_referencia` | opcional | texto | resumo padrão-ouro; habilita ROUGE/BLEU/chrF/BERTScore com referência. Sem ela, a transcrição vira pseudo-referência |
| `gt_qualidade_resumo` | por KPI | 1–5 | nota humana de qualidade do resumo |
| `gt_csat` | por KPI | 1–5 | CSAT (pesquisa real ou rótulo auditado) |
| `gt_razao_csat` | por KPI | texto | razão do CSAT validada |
| `gt_solicitacao_resolvida` | por KPI | `sim` \| `parcial` \| `nao` | |
| `gt_solver_reasoning` | por KPI | texto | explicação validada da resolução |
| `gt_razao_ligacao` | por KPI | rótulo da taxonomia | ver `config/kpis.yaml` |
| `gt_fcr` | por KPI | `sim` \| `nao` \| `indeterminado` | |
| `gt_empatia` | por KPI | 1–5 | |
| `gt_clareza_informacoes` | por KPI | 1–5 | |
| `gt_odc` | por KPI | código da taxonomia | ver `config/kpis.yaml` (substitua pela taxonomia oficial) |
| `prod_<kpi>` | opcional | idem `gt_` | saída do modelo hoje em produção → entra como baseline `producao_atual` |

Regras de leitura:
- Rótulos categóricos são normalizados (minúsculas, sem acento, espaços → `_`): `"Não"` → `nao`.
- Linha com `gt_<kpi>` vazio é ignorada **só naquele KPI**.
- Rótulo fora da taxonomia ou nota fora da escala gera aviso e é tratado como ausente.
- Colunas extras são ignoradas (ex.: `defeito_resumo`, `data_chamada` no exemplo).
- KPI sem coluna `gt_` é executado mas não avaliado (aviso no relatório).

Exemplo: `data/input/exemplo_kpis_validacao.csv` (120 chamadas sintéticas) e
`exemplo_kpis_validacao_5linhas.xlsx` (mesmo esquema em Excel).
