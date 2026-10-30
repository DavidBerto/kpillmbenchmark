"""Gera o arquivo tabular de EXEMPLO (dados sintéticos PT-BR, coerentes entre si).

Cada linha = uma chamada com:
  transcricao            diarizada (ATENDENTE:/CLIENTE:)
  resumo_referencia      resumo fiel (padrão-ouro)
  resumo                 resumo a ser avaliado, com defeitos controlados
  gt_*                   rótulos já validados (o que o benchmark usa como verdade)
  prod_*                 saídas do modelo hoje em produção (opcional; para baseline)
  defeito_resumo         tipo de defeito injetado (só para análise; não usado pelo benchmark)

Os rótulos são derivados das MESMAS variáveis latentes que geram a transcrição
(motivo, desfecho, empatia, clareza, reincidência), então são consistentes com o texto.

  python scripts/generate_sample_data.py --n 120 --out data/input/exemplo_kpis_validacao.csv
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

RNG_SEED = 20261020

NOMES = ["Ana", "Bruno", "Carla", "Diego", "Elaine", "Fábio", "Gabriela", "Henrique", "Isabela", "João",
         "Kátia", "Lucas", "Marina", "Nelson", "Olívia", "Paulo", "Renata", "Sérgio", "Tatiane", "Vinícius"]
SERVICOS = ["Música Plus", "Seguro Celular", "Banca Digital", "Games Club", "Proteção Premium"]
PLANOS = [("Controle 15GB", "Controle 25GB"), ("Pós 30GB", "Pós 50GB"), ("Fibra 300 Mega", "Fibra 600 Mega")]

ODC_TO_RESOLVIDA = {"resolvido_no_contato": "sim", "retencao_aplicada": "sim", "cancelamento_efetivado": "sim",
                    "protocolo_aberto": "parcial", "encaminhado_tecnico": "parcial",
                    "transferido": "nao", "sem_solucao": "nao"}

OUTCOMES = {
    "cobranca_indevida": [("resolvido_no_contato", .5), ("protocolo_aberto", .35), ("sem_solucao", .15)],
    "segunda_via_fatura": [("resolvido_no_contato", 1.0)],
    "cancelamento": [("retencao_aplicada", .4), ("cancelamento_efetivado", .45), ("transferido", .15)],
    "suporte_tecnico": [("resolvido_no_contato", .4), ("encaminhado_tecnico", .45), ("protocolo_aberto", .15)],
    "alteracao_plano": [("resolvido_no_contato", .75), ("sem_solucao", .25)],
    "portabilidade": [("protocolo_aberto", .6), ("transferido", .4)],
    "informacoes_gerais": [("resolvido_no_contato", 1.0)],
    "reclamacao_atendimento": [("protocolo_aberto", .5), ("transferido", .25), ("sem_solucao", .25)],
}
MOTIVO_WEIGHTS = {"cobranca_indevida": .2, "segunda_via_fatura": .12, "cancelamento": .14, "suporte_tecnico": .2,
                  "alteracao_plano": .1, "portabilidade": .08, "informacoes_gerais": .08, "reclamacao_atendimento": .08}


def pick(rng, items_weights):
    items, w = zip(*items_weights)
    w = np.array(w, float)
    return items[rng.choice(len(items), p=w / w.sum())]


def money(rng) -> str:
    return f"{rng.integers(9, 180)},{rng.choice([0, 50, 90, 99]):02d}"


# =========================================================================== cenário
def build_call(i: int, rng: np.random.Generator) -> dict:
    motivo = pick(rng, MOTIVO_WEIGHTS.items())
    odc = pick(rng, OUTCOMES[motivo])
    resolvida = ODC_TO_RESOLVIDA[odc]
    reincidente = rng.random() < 0.25
    empatia = int(pick(rng, [(1, .07), (2, .15), (3, .3), (4, .3), (5, .18)]))
    clareza = int(pick(rng, [(1, .06), (2, .16), (3, .28), (4, .3), (5, .2)]))
    nome = str(rng.choice(NOMES))
    v = {"nome": nome, "valor": money(rng), "servico": str(rng.choice(SERVICOS)),
         "protocolo": "".join(str(d) for d in rng.integers(0, 10, 11)), "prazo": int(rng.choice([3, 5, 7, 10])),
         "dia": int(rng.integers(5, 28)), "mes": int(rng.integers(1, 12))}
    v["plano_de"], v["plano_para"] = PLANOS[rng.integers(len(PLANOS))]

    S = scenario(motivo, odc, v, clareza)

    # ---------------- transcrição
    T = []
    greet = {1: "Central de atendimento. Fala.", 2: "Central de atendimento, informe o CPF do titular.",
             3: f"Central de atendimento, meu nome é Paula, com quem eu falo?",
             4: f"Olá, central de atendimento, meu nome é Paula, com quem eu falo, por favor?",
             5: f"Bom dia! Aqui é a Paula, é um prazer te atender. Com quem eu falo?"}[empatia]
    T.append(f"ATENDENTE: {greet}")
    T.append(f"CLIENTE: É {nome}. {S['cliente']}")
    if reincidente:
        T.append("CLIENTE: E olha, essa já é a terceira vez que eu ligo por causa disso, ninguém resolve.")
    T.append("ATENDENTE: " + {1: "Tá, já entendi, aguarda aí.", 2: "Certo. Aguarde que vou verificar.",
                              3: "Entendi. Vou verificar no sistema, um momento.",
                              4: "Sinto muito pelo transtorno, vou verificar agora mesmo pra você.",
                              5: f"{nome}, eu entendo totalmente a sua chateação e peço desculpas. Vou cuidar disso "
                                 "pessoalmente, tudo bem?"}[empatia])
    T.append("CLIENTE: Tá bom.")
    T += [f"ATENDENTE: {l}" for l in S["acoes"]]
    T.append("ATENDENTE: " + S["explicacao"])
    if clareza == 5:
        T.append("ATENDENTE: Ficou claro pra você? Quer que eu repita o protocolo?")
        T.append("CLIENTE: Ficou claro, obrigado.")

    base = {"sim": 4.3, "parcial": 3.0, "nao": 1.7}[resolvida]
    csat_lat = base + 0.35 * (empatia - 3) + 0.2 * (clareza - 3) - 0.7 * reincidente + rng.normal(0, 0.45)
    csat = int(np.clip(round(csat_lat), 1, 5))
    reacao = {1: "Isso é um absurdo, vou registrar reclamação no Procon.", 2: "Não resolveu nada, que decepção.",
              3: "Tá, vou aguardar então.", 4: "Ok, obrigado pela ajuda.",
              5: "Nossa, muito obrigado, você foi ótima!"}[csat]
    T.append(f"CLIENTE: {reacao}")
    T.append("ATENDENTE: " + ("Mais alguma coisa? Tchau." if empatia <= 2 else
                              "Agradeço o contato, tenha um ótimo dia!"))
    transcricao = "\n".join(T)

    # ---------------- resumo de referência (fiel) + rótulos de texto
    postura = posture_sentence(empatia, clareza)
    reacao_s = {1: "Cliente ficou muito irritado e ameaçou reclamar no Procon.",
                2: "Cliente demonstrou insatisfação com o atendimento.", 3: "Cliente aceitou aguardar.",
                4: "Cliente agradeceu.", 5: "Cliente agradeceu e elogiou a atendente."}[csat]
    reinc_s = "Cliente relatou que já havia ligado outras vezes pelo mesmo motivo." if reincidente else None

    def make(variant: int, drop: set[str] = frozenset(), override: dict | None = None) -> str:
        parts = {"problema": S["problema"][variant], "reinc": reinc_s, "acao": S["acao"][variant],
                 "proximo": S["proximo"][variant] if S["proximo"] else None, "postura": postura[variant],
                 "reacao": reacao_s}
        if override:
            parts.update(override)
        order = ["problema", "reinc", "acao", "proximo", "postura", "reacao"]
        return " ".join(parts[k] for k in order if parts.get(k) and k not in drop)

    resumo_ref = make(0)

    # ---------------- resumo avaliado com defeito controlado
    q = int(pick(rng, [(5, .3), (4, .25), (3, .2), (2, .15), (1, .1)]))
    if q == 5:
        resumo, defeito = make(1), "nenhum"
    elif q == 4:
        drop = {"postura"} if rng.random() < 0.5 or not S["proximo"] else {"proximo"}
        resumo, defeito = make(1, drop=drop), f"omissao_menor:{next(iter(drop))}"
    elif q == 3:
        resumo, defeito = make(1, drop={"acao", "proximo"}), "omissao_desfecho"
    elif q == 2:
        resumo, defeito = hallucinate(make(1), v, rng), "alucinacao_numerica"
    else:
        resumo = hallucinate(make(1, override={"acao": S["desfecho_errado"], "proximo": None}), v, rng)
        defeito = "desfecho_errado+alucinacao"
    nota_humana = int(np.clip(q + (rng.choice([-1, 1]) if rng.random() < 0.12 else 0), 1, 5))

    fcr = "sim" if resolvida == "sim" and not reincidente else "nao"
    razao_csat = csat_reason(csat, resolvida, motivo, odc, empatia, reincidente)

    row = {
        "call_id": f"CALL-{i:06d}", "data_chamada": f"2026-{v['mes']:02d}-{v['dia']:02d}",
        "transcricao": transcricao, "resumo": resumo, "resumo_referencia": resumo_ref,
        "gt_qualidade_resumo": nota_humana, "gt_csat": csat, "gt_razao_csat": razao_csat,
        "gt_solicitacao_resolvida": resolvida, "gt_solver_reasoning": S["solver"],
        "gt_razao_ligacao": motivo, "gt_fcr": fcr, "gt_empatia": empatia,
        "gt_clareza_informacoes": clareza, "gt_odc": odc, "defeito_resumo": defeito,
    }
    row.update(production_outputs(row, rng))
    return row


def posture_sentence(emp: int, cla: int) -> tuple[str, str]:
    e = {1: ("A atendente foi ríspida", "Atendente tratou o cliente de forma ríspida"),
         2: ("A atendente foi fria e protocolar", "Atendimento frio, apenas protocolar"),
         3: ("A atendente foi cordial", "Atendimento cordial e objetivo"),
         4: ("A atendente demonstrou empatia e pediu desculpas pelo transtorno",
             "Atendente se desculpou pelo transtorno e mostrou empatia"),
         5: ("A atendente acolheu o cliente, assumiu o caso pessoalmente e o tranquilizou",
             "Atendente foi muito acolhedora, assumiu o caso e tranquilizou o cliente")}[emp]
    c = {1: ("e passou informações contraditórias.", "e as informações passadas foram contraditórias."),
         2: ("e deu informações vagas, sem prazo.", "e as orientações foram vagas, sem prazo definido."),
         3: ("e explicou o procedimento, sem informar o prazo.", "e explicou o procedimento, mas não deu prazo."),
         4: ("e explicou o procedimento e o próximo passo.", "e orientou sobre o procedimento e o próximo passo."),
         5: ("e explicou com clareza o procedimento, o prazo e o protocolo, confirmando o entendimento do cliente.",
             "e detalhou procedimento, prazo e protocolo, checando se o cliente entendeu.")}[cla]
    return (f"{e[0]} {c[0]}", f"{e[1]} {c[1]}")


def scenario(motivo: str, odc: str, v: dict, clareza: int) -> dict:
    prazo_txt = f"em até {v['prazo']} dias úteis" if clareza >= 4 else ("em breve" if clareza <= 2 else "")
    prot_txt = f"O protocolo é {v['protocolo']}." if clareza >= 4 else ""
    if clareza == 1:
        prot_txt = "Na verdade pode ser que já esteja resolvido, ou não, depende do sistema."
    prox_ok = clareza >= 4

    if motivo == "cobranca_indevida":
        cliente = f"Apareceu na minha fatura uma cobrança de R$ {v['valor']} de um tal de {v['servico']} que eu nunca contratei."
        problema = (f"Cliente contestou cobrança de R$ {v['valor']} referente ao serviço {v['servico']}, que afirma não ter contratado.",
                    f"Cliente ligou para contestar a cobrança de R$ {v['valor']} do serviço {v['servico']}, não reconhecido por ele.")
        if odc == "resolvido_no_contato":
            acoes = ["Localizei aqui, o serviço foi ativado sem confirmação.", f"Já cancelei o {v['servico']} e lancei o estorno de R$ {v['valor']} na próxima fatura."]
            acao = (f"A atendente cancelou o serviço e aplicou estorno de R$ {v['valor']} na próxima fatura.",
                    f"Serviço cancelado e estorno de R$ {v['valor']} lançado para a próxima fatura.")
            solver = f"Atendente identificou ativação indevida do {v['servico']}, cancelou o serviço e lançou estorno de R$ {v['valor']} na próxima fatura, resolvendo na ligação."
            errado = "A atendente informou que a cobrança é devida e não haverá estorno."
        elif odc == "protocolo_aberto":
            acoes = ["Vou precisar abrir uma contestação para a área financeira analisar."]
            acao = ("A atendente abriu contestação para análise da área financeira.",
                    "Foi registrada contestação para análise do financeiro.")
            solver = f"Atendente não pôde estornar na hora; abriu contestação para análise do financeiro{(' com prazo de ' + str(v['prazo']) + ' dias úteis') if prox_ok else ''}."
            errado = f"A atendente estornou imediatamente o valor de R$ {v['valor']}."
        else:
            acoes = ["Infelizmente o sistema mostra que o serviço foi contratado, não consigo fazer nada."]
            acao = ("A atendente afirmou que o serviço consta como contratado e não realizou estorno.",
                    "Atendente disse que o serviço aparece como contratado e não estornou o valor.")
            solver = "Atendente alegou que o serviço consta como contratado e encerrou sem estorno nem encaminhamento; demanda não resolvida."
            errado = f"A atendente cancelou o serviço e estornou R$ {v['valor']}."
        explic = f"{('O estorno/análise ocorre ' + prazo_txt + '. ') if prazo_txt else ''}{prot_txt}".strip() or "É isso."
        proximo = (f"Prazo de {v['prazo']} dias úteis, protocolo {v['protocolo']}.",
                   f"Protocolo {v['protocolo']} com prazo de {v['prazo']} dias úteis.") if prox_ok and odc != "sem_solucao" else None

    elif motivo == "segunda_via_fatura":
        cliente = f"Eu preciso da segunda via da fatura que vence dia {v['dia']}, perdi o boleto."
        problema = (f"Cliente solicitou a segunda via da fatura com vencimento no dia {v['dia']}.",
                    f"Cliente pediu segunda via da fatura que vence no dia {v['dia']}.")
        acoes = ["Vou te enviar agora por SMS e e-mail, com código de barras e PIX."]
        acao = ("A atendente enviou a segunda via por SMS e e-mail, com código de barras e PIX.",
                "Segunda via enviada por SMS e e-mail com opção de PIX.")
        solver = "Atendente enviou a segunda via por SMS e e-mail com código de barras e PIX, resolvendo a demanda na ligação."
        errado = "A atendente não conseguiu emitir a segunda via."
        explic = "Já está no seu celular." if clareza < 4 else f"Enviei agora; o valor é R$ {v['valor']} e o PIX compensa na hora."
        proximo = (f"Valor da fatura informado: R$ {v['valor']}.", f"Fatura de R$ {v['valor']}.") if prox_ok else None

    elif motivo == "cancelamento":
        cliente = f"Eu quero cancelar o meu plano {v['plano_de']}, está muito caro."
        problema = (f"Cliente solicitou o cancelamento do plano {v['plano_de']} alegando preço alto.",
                    f"Cliente ligou pedindo para cancelar o {v['plano_de']} por considerar o valor alto.")
        if odc == "retencao_aplicada":
            acoes = [f"Antes de cancelar, consigo te oferecer 30% de desconto por 6 meses no {v['plano_de']}.", "Cliente aceitou a oferta."]
            acao = ("A atendente ofereceu 30% de desconto por 6 meses e o cliente aceitou permanecer.",
                    "Cliente aceitou oferta de retenção com 30% de desconto por 6 meses.")
            solver = "Atendente aplicou oferta de retenção de 30% de desconto por 6 meses, aceita pelo cliente, que manteve o plano."
            errado = "O plano foi cancelado na ligação."
        elif odc == "cancelamento_efetivado":
            acoes = ["Ofereci desconto mas o senhor recusou, então efetivei o cancelamento."]
            acao = ("O cliente recusou a oferta de retenção e a atendente efetivou o cancelamento.",
                    "Retenção recusada; cancelamento concluído pela atendente.")
            solver = "Atendente ofereceu retenção, recusada; efetivou o cancelamento conforme pedido do cliente."
            errado = "O cliente aceitou a oferta de desconto e manteve o plano."
        else:
            acoes = ["Vou te transferir para o setor de retenção."]
            acao = ("A atendente transferiu a ligação para o setor de retenção.",
                    "Ligação transferida para a retenção.")
            solver = "Atendente não concluiu o cancelamento e transferiu o cliente para o setor de retenção."
            errado = "O cancelamento foi concluído pela atendente."
        explic = (f"O protocolo é {v['protocolo']}." if clareza >= 4 else "Pronto.")
        proximo = (f"Protocolo {v['protocolo']}.", f"Protocolo informado: {v['protocolo']}.") if prox_ok and odc != "transferido" else None

    elif motivo == "suporte_tecnico":
        cliente = "Minha internet está caindo toda hora desde ontem, não consigo trabalhar."
        problema = ("Cliente relatou quedas frequentes da internet desde o dia anterior.",
                    "Cliente reclamou de instabilidade e quedas da conexão desde ontem.")
        if odc == "resolvido_no_contato":
            acoes = ["Fiz um reset remoto no seu modem e reconfigurei o sinal.", "Pode testar agora?", "Cliente confirmou que voltou."]
            acao = ("A atendente fez reset remoto do modem e o cliente confirmou a normalização.",
                    "Após reset remoto do modem, a conexão foi normalizada e confirmada pelo cliente.")
            solver = "Atendente realizou reset remoto do modem e reconfiguração; cliente testou e confirmou a normalização durante a ligação."
            errado = "Foi necessário agendar visita técnica, pois o reset remoto não funcionou."
        elif odc == "encaminhado_tecnico":
            acoes = ["Os testes indicam problema no cabo externo.", f"Agendei visita técnica para o dia {v['dia']}/{v['mes']:02d}."]
            acao = (f"A atendente identificou falha no cabeamento externo e agendou visita técnica para {v['dia']}/{v['mes']:02d}.",
                    f"Testes apontaram problema no cabo externo; visita técnica marcada para {v['dia']}/{v['mes']:02d}.")
            solver = f"Atendente diagnosticou falha no cabo externo e agendou visita técnica para {v['dia']}/{v['mes']:02d}; resolução depende da visita."
            errado = "O problema foi resolvido remotamente durante a ligação."
        else:
            acoes = ["Vou abrir um chamado para a engenharia analisar a sua região."]
            acao = ("A atendente abriu chamado para análise da engenharia na região.",
                    "Chamado aberto para a engenharia verificar a região do cliente.")
            solver = "Atendente abriu chamado para análise de rede na região; sem solução imediata."
            errado = "A conexão foi normalizada durante a ligação."
        explic = f"{('Prazo ' + prazo_txt + '. ') if prazo_txt else ''}{prot_txt}".strip() or "Aguarde."
        proximo = (f"Protocolo {v['protocolo']}, prazo de {v['prazo']} dias úteis.",
                   f"Prazo de {v['prazo']} dias úteis; protocolo {v['protocolo']}.") if prox_ok and odc != "resolvido_no_contato" else None

    elif motivo == "alteracao_plano":
        cliente = f"Quero mudar do {v['plano_de']} para o {v['plano_para']}."
        problema = (f"Cliente pediu a troca do plano {v['plano_de']} para o {v['plano_para']}.",
                    f"Cliente solicitou migrar do {v['plano_de']} para o {v['plano_para']}.")
        if odc == "resolvido_no_contato":
            acoes = [f"Feito, migrei para o {v['plano_para']}, a nova mensalidade é R$ {v['valor']}."]
            acao = (f"A atendente realizou a migração para o {v['plano_para']}, com mensalidade de R$ {v['valor']}.",
                    f"Migração para o {v['plano_para']} concluída; nova mensalidade de R$ {v['valor']}.")
            solver = f"Atendente efetuou a migração para o {v['plano_para']} na ligação e informou a nova mensalidade de R$ {v['valor']}."
            errado = "A migração não foi possível por falta de cobertura."
        else:
            acoes = [f"Infelizmente o {v['plano_para']} não está disponível no seu endereço."]
            acao = (f"A atendente informou que o {v['plano_para']} não está disponível no endereço do cliente.",
                    f"O {v['plano_para']} não tem disponibilidade no endereço do cliente.")
            solver = f"Atendente verificou que o {v['plano_para']} não está disponível no endereço; troca não realizada."
            errado = f"A atendente concluiu a migração para o {v['plano_para']}."
        explic = "Vale a partir do próximo ciclo." if clareza >= 3 else "Depois o sistema atualiza."
        proximo = ("A alteração vale a partir do próximo ciclo de faturamento.",
                   "Mudança válida a partir do próximo ciclo.") if prox_ok and odc == "resolvido_no_contato" else None

    elif motivo == "portabilidade":
        cliente = "Quero trazer meu número de outra operadora para vocês."
        problema = ("Cliente solicitou portabilidade do número vindo de outra operadora.",
                    "Cliente quer portar seu número de outra operadora.")
        if odc == "protocolo_aberto":
            acoes = ["Registrei o pedido de portabilidade."]
            acao = ("A atendente registrou o pedido de portabilidade.", "Pedido de portabilidade registrado pela atendente.")
            solver = "Atendente registrou a solicitação de portabilidade, que depende da janela regulatória para ser concluída."
            errado = "A portabilidade foi concluída na hora."
        else:
            acoes = ["Isso é com o setor de vendas, vou transferir."]
            acao = ("A atendente transferiu o cliente para o setor de vendas.", "Ligação transferida para vendas.")
            solver = "Atendente não tratou a portabilidade e transferiu o cliente para o setor de vendas."
            errado = "A atendente registrou e concluiu a portabilidade."
        explic = (f"A portabilidade ocorre {prazo_txt}. {prot_txt}".strip() if prazo_txt else "Aguarde o SMS.")
        proximo = (f"Prazo de {v['prazo']} dias úteis, protocolo {v['protocolo']}.",
                   f"Protocolo {v['protocolo']}; conclusão em até {v['prazo']} dias úteis.") if prox_ok and odc == "protocolo_aberto" else None

    elif motivo == "informacoes_gerais":
        cliente = "Só queria saber até que horas a loja do shopping fica aberta e se lá faz troca de chip."
        problema = ("Cliente pediu informações sobre o horário da loja do shopping e troca de chip.",
                    "Cliente perguntou o horário da loja do shopping e se ela faz troca de chip.")
        acoes = ["A loja abre das 10h às 22h e faz troca de chip, sim."]
        acao = ("A atendente informou o horário de funcionamento (10h às 22h) e confirmou a troca de chip.",
                "Atendente informou funcionamento das 10h às 22h e que a loja troca chip.")
        solver = "Atendente informou horário da loja (10h às 22h) e confirmou que é feita troca de chip, esclarecendo a dúvida."
        errado = "A atendente não soube informar o horário da loja."
        explic = "Leve um documento com foto." if clareza >= 4 else "É isso."
        proximo = ("Orientou levar documento com foto.", "Recomendou levar documento com foto.") if prox_ok else None

    else:  # reclamacao_atendimento
        cliente = "Quero reclamar do técnico que veio ontem, chegou atrasado e foi grosseiro."
        problema = ("Cliente reclamou de atraso e grosseria do técnico na visita do dia anterior.",
                    "Cliente registrou queixa sobre o técnico, que atrasou e foi grosseiro na visita de ontem.")
        if odc == "protocolo_aberto":
            acoes = ["Registrei a sua reclamação para a ouvidoria de campo."]
            acao = ("A atendente registrou a reclamação para a ouvidoria de campo.", "Reclamação registrada na ouvidoria de campo.")
            solver = "Atendente registrou a reclamação na ouvidoria de campo para apuração; sem resposta imediata."
            errado = "O técnico foi advertido e o caso encerrado na ligação."
        elif odc == "transferido":
            acoes = ["Vou transferir para a ouvidoria."]
            acao = ("A atendente transferiu a ligação para a ouvidoria.", "Ligação transferida para a ouvidoria.")
            solver = "Atendente não registrou a reclamação e transferiu para a ouvidoria."
            errado = "A reclamação foi registrada e resolvida na ligação."
        else:
            acoes = ["Não tenho como registrar reclamação de técnico por aqui."]
            acao = ("A atendente disse não ser possível registrar a reclamação por esse canal.",
                    "Atendente informou que não consegue registrar esse tipo de reclamação.")
            solver = "Atendente informou não ser possível registrar a reclamação pelo canal e não ofereceu alternativa."
            errado = "A atendente registrou a reclamação e abriu protocolo."
        explic = f"{('Retorno ' + prazo_txt + '. ') if prazo_txt else ''}{prot_txt}".strip() or "Pronto."
        proximo = (f"Protocolo {v['protocolo']}, retorno em até {v['prazo']} dias úteis.",
                   f"Retorno previsto em {v['prazo']} dias úteis; protocolo {v['protocolo']}.") if prox_ok and odc == "protocolo_aberto" else None

    return {"cliente": cliente, "problema": problema, "acoes": acoes, "acao": acao, "explicacao": explic,
            "proximo": proximo, "solver": solver, "desfecho_errado": errado}


def hallucinate(text: str, v: dict, rng: np.random.Generator) -> str:
    """Troca valores/protocolos/datas por números que NÃO estão na transcrição."""
    out = text
    if v["valor"] in out:
        out = out.replace(v["valor"], money(rng))
    if v["protocolo"] in out:
        out = out.replace(v["protocolo"], "".join(str(d) for d in rng.integers(0, 10, 11)))
    m = re.search(r"(\d+) dias úteis", out)
    if m:
        out = out.replace(m.group(0), f"{int(m.group(1)) + int(rng.choice([2, 5]))} dias úteis", 1)
    if out == text:  # nada numérico: inventa um compromisso
        out = text + f" A atendente garantiu bônus de R$ {money(rng)} na próxima fatura."
    return out


def csat_reason(csat: int, resolvida: str, motivo: str, odc: str, emp: int, reinc: bool) -> str:
    fator = {"sim": "a demanda foi resolvida na ligação", "parcial": "a demanda ficou encaminhada, sem solução imediata",
             "nao": "a demanda não foi resolvida"}[resolvida]
    pos, neg = [], []
    (pos if resolvida == "sim" else neg).append(fator)
    if reinc:
        neg.append("o cliente já havia ligado outras vezes pelo mesmo problema")
    if emp >= 4:
        pos.append("a atendente foi empática")
    elif emp <= 2:
        neg.append("a atendente foi pouco cordial")
    if csat >= 4:
        s = f"Cliente satisfeito porque {' e '.join(pos or ['o atendimento foi conduzido sem atritos'])}"
        return s + (f", mas {' e '.join(neg)}." if neg else ".")
    if csat == 3:
        return f"Satisfação neutra: {' e '.join(neg or pos)}" + (f", mas {' e '.join(pos)}." if neg and pos else ".")
    s = f"Cliente insatisfeito porque {' e '.join(neg or [fator])}"
    return s + (f", mas {' e '.join(pos)}." if pos and neg else ".")


def production_outputs(row: dict, rng: np.random.Generator) -> dict:
    """Simula o modelo hoje em produção (acerto ~72%)."""
    out = {}
    cats = {"solicitacao_resolvida": ["sim", "parcial", "nao"], "first_call_resolution": ["sim", "nao", "indeterminado"],
            "razao_ligacao": list(MOTIVO_WEIGHTS), "odc": list(ODC_TO_RESOLVIDA)}
    gtcol = {"solicitacao_resolvida": "gt_solicitacao_resolvida", "first_call_resolution": "gt_fcr",
             "razao_ligacao": "gt_razao_ligacao", "odc": "gt_odc"}
    for k, labels in cats.items():
        gt = row[gtcol[k]]
        out[f"prod_{k}"] = gt if rng.random() < 0.72 else str(rng.choice([l for l in labels if l != gt]))
    for k, col in {"csat": "gt_csat", "empatia": "gt_empatia", "clareza_informacoes": "gt_clareza_informacoes",
                   "qualidade_resumo": "gt_qualidade_resumo"}.items():
        gt = row[col]
        out[f"prod_{k}"] = gt if rng.random() < 0.6 else int(np.clip(gt + rng.choice([-1, 1]), 1, 5))
    return out


def main(n: int = 120, out: Path | str = "data/input/exemplo_kpis_validacao.csv") -> Path:
    rng = np.random.default_rng(RNG_SEED)
    df = pd.DataFrame([build_call(i + 1, rng) for i in range(n)])
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8")
    df.head(5).to_excel(out.with_name("exemplo_kpis_validacao_5linhas.xlsx"), index=False)
    print(f"{len(df)} chamadas -> {out}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "data/input/exemplo_kpis_validacao.csv"))
    a = ap.parse_args()
    main(a.n, a.out)
