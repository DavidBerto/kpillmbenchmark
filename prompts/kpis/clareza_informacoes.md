### SYSTEM
{% include "_base_system.md" %}

KPI: CLAREZA DAS INFORMAÇÕES
Classifique a clareza das informações passadas pelo atendente ao cliente.

Valores permitidos:
{% for k, v in labels.items() %}- "{{ k }}": {{ v }}
{% endfor %}
Critérios:
- "sim" quando o atendente comunica com clareza, informa o próximo passo, prazo e procedimento de forma compreensível.
- "nao" quando as informações são vagas, contraditórias, incompletas ou deixam o cliente sem saber o que fazer.
- Evidência insuficiente: "nao".

Formato de saída (JSON):
{"justificativa": "<até 40 palavras>", "clareza": "<{{ labels.keys() | join('|') }}>"}

### USER
<resumo>
{{ resumo }}
</resumo>
