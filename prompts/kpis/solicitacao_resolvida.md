### SYSTEM
{% include "_base_system.md" %}

KPI: SOLICITAÇÃO FOI RESOLVIDA
Classifique se a demanda PRINCIPAL do cliente foi resolvida.

Valores permitidos:
{% for k, v in labels.items() %}- "{{ k }}": {{ v }}
{% endfor %}
Critérios:
- Avalie a demanda principal, não pedidos secundários.
- Promessa vaga ("vamos verificar") sem protocolo ou prazo não conta como "parcial": é "nao".
- Evidência insuficiente: "nao".

Formato de saída (JSON):
{"justificativa": "<até 40 palavras>", "resolvida": "<{{ labels.keys() | join('|') }}>"}

### USER
<resumo>
{{ resumo }}
</resumo>
