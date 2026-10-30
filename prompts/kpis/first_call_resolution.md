### SYSTEM
{% include "_base_system.md" %}

KPI: FIRST CALL RESOLUTION (FCR)
Classifique se a demanda foi resolvida no primeiro contato, sem necessidade de o cliente
voltar a ligar sobre o mesmo assunto.

Valores permitidos:
{% for k, v in labels.items() %}- "{{ k }}": {{ v }}
{% endfor %}
Critérios:
- "sim" somente se a demanda foi resolvida DURANTE esta ligação e não há menção de que o
  cliente já ligou antes sobre o mesmo problema.
- Encaminhamento (protocolo, análise, visita técnica), transferência ou demanda não
  resolvida -> "nao".
- Cliente reincidente no mesmo assunto -> "nao", mesmo que resolvido agora.
- Use "indeterminado" somente quando o resumo não permitir decidir.

Formato de saída (JSON):
{"justificativa": "<até 40 palavras>", "fcr": "<{{ labels.keys() | join('|') }}>"}

### USER
<resumo>
{{ resumo }}
</resumo>
