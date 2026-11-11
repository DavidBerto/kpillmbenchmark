### SYSTEM
{% include "_base_system.md" %}

KPI: EMPATIA DO ATENDENTE
Classifique a empatia demonstrada pelo atendente.

Valores permitidos:
{% for k, v in labels.items() %}- "{{ k }}": {{ v }}
{% endfor %}
Critérios:
- "sim" quando o atendente reconhece a frustração, acolhe o cliente, pede desculpas quando cabível e transmite cuidado.
- "nao" quando o atendente é frio, protocolar, ignora a emoção do cliente ou não demonstra acolhimento.
- Evidência insuficiente: "nao".

Formato de saída (JSON):
{"justificativa": "<até 40 palavras>", "empatia": "<{{ labels.keys() | join('|') }}>"}

### USER
<resumo>
{{ resumo }}
</resumo>
