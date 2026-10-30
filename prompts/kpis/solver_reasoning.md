### SYSTEM
{% include "_base_system.md" %}

KPI: SOLVER REASONING
Descreva como a demanda principal do cliente foi (ou não foi) resolvida: qual ação o
atendente tomou, qual foi o resultado e, se houver, qual o próximo passo (protocolo, prazo,
visita técnica, estorno).

Requisitos:
- Encadeie ação -> resultado -> próximo passo.
- Cite valores, prazos e protocolos somente se estiverem no resumo.
- Se não houve resolução, diga o que impediu.
Evidência insuficiente: "Resumo não descreve a resolução."

Formato de saída (JSON):
{"solver_reasoning": "<até 40 palavras>"}

### USER
<resumo>
{{ resumo }}
</resumo>
