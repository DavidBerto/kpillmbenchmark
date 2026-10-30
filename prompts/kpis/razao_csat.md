### SYSTEM
{% include "_base_system.md" %}

KPI: RAZÃO DO CSAT
Explique, em uma ou duas frases, o principal motivo que determina o nível de satisfação do
cliente neste atendimento.

A razão deve:
- citar o fator decisivo (ex.: demanda resolvida/não resolvida, cobrança estornada,
  necessidade de novo contato, demora, transferências, postura do atendente);
- ser sustentada pelo resumo, sem inventar fatos;
- indicar a direção (o que gerou satisfação ou insatisfação).
Evidência insuficiente: "Resumo não traz elementos suficientes para explicar a satisfação."

Formato de saída (JSON):
{"razao_csat": "<até 40 palavras>"}

### USER
<resumo>
{{ resumo }}
</resumo>
