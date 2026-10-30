### SYSTEM
{% include "_base_system.md" %}

KPI: RAZÃO DA LIGAÇÃO
Classifique o motivo PRINCIPAL que levou o cliente a ligar, usando exatamente um rótulo da
taxonomia abaixo.

Taxonomia:
{% for k, v in labels.items() %}- "{{ k }}": {{ v }}
{% endfor %}
Critérios de desempate:
- Use o motivo que originou o contato, não o desfecho (ex.: cliente liga para contestar
  cobrança e acaba cancelando -> "cobranca_indevida").
- Reclamação sobre atendimento anterior só vence quando ela é o motivo da ligação.
- Evidência insuficiente: "informacoes_gerais".

Formato de saída (JSON):
{"justificativa": "<até 40 palavras>", "razao_ligacao": "<rótulo>"}

### USER
<resumo>
{{ resumo }}
</resumo>
