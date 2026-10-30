### SYSTEM
{% include "_base_system.md" %}

KPI: CSAT INFERIDO
Estime a nota de satisfação que o cliente daria ao final deste atendimento, numa escala
de {{ scale.min }} a {{ scale.max }}, com base no resumo da chamada.

Âncoras da escala:
- 1: cliente irritado ou frustrado, demanda não atendida, ameaça cancelar/reclamar em órgão.
- 2: insatisfeito; demanda não atendida ou atendida com muito atrito.
- 3: neutro; demanda parcialmente atendida ou sem sinais claros de emoção.
- 4: satisfeito; demanda atendida com algum atrito ou espera.
- 5: muito satisfeito; demanda atendida e cliente agradece/elogia.

Considere, nesta ordem: desfecho da demanda, sinais emocionais do cliente,
esforço exigido do cliente (repetições, transferências, espera), postura do atendente.
Evidência insuficiente: use 3.

Formato de saída (JSON):
{"justificativa": "<até 40 palavras>", "csat": <inteiro {{ scale.min }}-{{ scale.max }}>}

### USER
<resumo>
{{ resumo }}
</resumo>
