### SYSTEM
{% include "_base_system.md" %}

KPI: EMPATIA DO ATENDENTE
Avalie a empatia demonstrada pelo atendente, de {{ scale.min }} a {{ scale.max }}.

Âncoras da escala:
- 1: ríspido, desrespeitoso ou ignora a emoção do cliente.
- 2: frio, protocolar, sem reconhecer o problema do cliente.
- 3: cordial, mas genérico; reconhecimento mínimo.
- 4: reconhece o problema e a emoção do cliente, pede desculpas quando cabível.
- 5: acolhe ativamente, personaliza, assume responsabilidade e tranquiliza o cliente.

Avalie comportamento do atendente, NÃO o desfecho: um caso não resolvido pode ter empatia 5.
Evidência insuficiente (resumo não descreve a postura): 3.

Formato de saída (JSON):
{"justificativa": "<até 40 palavras>", "empatia": <inteiro {{ scale.min }}-{{ scale.max }}>}

### USER
<resumo>
{{ resumo }}
</resumo>
