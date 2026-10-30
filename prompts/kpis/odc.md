### SYSTEM
{% include "_base_system.md" %}

KPI: ODC (código de desfecho do contato)
Classifique o desfecho operacional do atendimento com exatamente um código da taxonomia.

Taxonomia:
{% for k, v in labels.items() %}- "{{ k }}": {{ v }}
{% endfor %}
Critérios de desempate:
- Use o ÚLTIMO desfecho efetivo da chamada.
- Retenção aceita prevalece sobre protocolo aberto na mesma chamada.
- Evidência insuficiente: "sem_solucao".

Formato de saída (JSON):
{"justificativa": "<até 40 palavras>", "odc": "<código>"}

### USER
<resumo>
{{ resumo }}
</resumo>
