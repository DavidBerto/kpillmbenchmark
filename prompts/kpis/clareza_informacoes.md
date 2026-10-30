### SYSTEM
{% include "_base_system.md" %}

KPI: CLAREZA DAS INFORMAÇÕES
Avalie a clareza das informações passadas pelo atendente ao cliente, de {{ scale.min }} a {{ scale.max }}.

Âncoras da escala:
- 1: informações erradas, contraditórias ou ausentes; cliente sai sem saber o que fazer.
- 2: informações vagas ("vamos verificar") sem prazo, valor ou procedimento.
- 3: informação correta, porém incompleta (falta prazo, protocolo ou próximo passo).
- 4: explica o procedimento e o próximo passo, com pequenas lacunas.
- 5: explica com precisão o quê, quanto, quando e como; confirma o entendimento do cliente.

Evidência insuficiente: 3.

Formato de saída (JSON):
{"justificativa": "<até 40 palavras>", "clareza": <inteiro {{ scale.min }}-{{ scale.max }}>}

### USER
<resumo>
{{ resumo }}
</resumo>
