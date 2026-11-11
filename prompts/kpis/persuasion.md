### SYSTEM
{% include "_base_system.md" %}

KPI: PERSUASÃO / RETENÇÃO
Classifique se o atendente tentou persuadir ou reter o cliente para manter o produto/serviço.

Valores permitidos:
{% for k, v in labels.items() %}- "{{ k }}": {{ v }}
{% endfor %}
Critérios:
- "sim" quando houver menção explícita de incentivo, retenção, oferta de manutenção, permanência, renegociação ou tentativa de convencer o cliente a continuar com o produto/serviço.
- "nao" quando não houver qualquer tentativa de persuasão, retenção ou incentivo.
- Evidência insuficiente: "nao".

Formato de saída (JSON):
{"justificativa": "<até 40 palavras>", "persuasion": "<{{ labels.keys() | join('|') }}>"}

### USER
<resumo>
{{ resumo }}
</resumo>
