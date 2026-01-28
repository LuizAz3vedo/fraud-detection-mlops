"""
Demo: Cliente que simula um sistema de pagamentos usando a API de fraude.

Este script mostra como um sistema real integraria com a API.

Uso:
    1. Rode a API: make api
    2. Em outro terminal: python demo/client.py
"""

import requests
import random
import time
from datetime import datetime


API_URL = "http://localhost:8000"


def check_fraud(transaction: dict) -> dict:
    """
    Envia transacao para API de fraude.

    Args:
        transaction: Dados da transacao

    Returns:
        Resultado da analise
    """
    response = requests.post(f"{API_URL}/predict", json=transaction)
    return response.json()


def simulate_transaction():
    """Simula uma transacao aleatoria."""
    # 90% transacoes normais, 10% suspeitas
    is_suspicious = random.random() < 0.1

    if is_suspicious:
        # Transacao suspeita: valor alto, features anomalas
        return {
            "amount": random.uniform(2000, 10000),
            "v14": random.uniform(-10, -5),  # v14 muito negativo = suspeito
            "v10": random.uniform(-5, -3),
            "v4": random.uniform(3, 6),
        }
    else:
        # Transacao normal
        return {
            "amount": random.uniform(10, 500),
            "v14": random.uniform(-2, 2),
            "v10": random.uniform(-1, 1),
            "v4": random.uniform(-1, 2),
        }


def process_payment(customer: str, amount: float, transaction: dict):
    """
    Simula processamento de pagamento com verificacao de fraude.

    Este e o fluxo que um sistema real usaria.
    """
    print(f"\n{'='*60}")
    print(f"PROCESSANDO PAGAMENTO")
    print(f"{'='*60}")
    print(f"Cliente: {customer}")
    print(f"Valor: R$ {amount:,.2f}")
    print(f"Horario: {datetime.now().strftime('%H:%M:%S')}")

    # 1. Chamar API de fraude
    print(f"\n[1] Verificando fraude...")
    result = check_fraud(transaction)

    prob = result['fraud_probability']
    risk = result['risk_level']
    is_fraud = result['is_fraud']

    print(f"    Probabilidade: {prob:.1%}")
    print(f"    Nivel de risco: {risk.upper()}")

    # 2. Decidir com base no resultado
    print(f"\n[2] Decisao:")

    if is_fraud or prob > 0.7:
        print(f"    BLOQUEADO - Suspeita de fraude!")
        print(f"    Acao: Enviar SMS de verificacao ao cliente")
        return "BLOCKED"

    elif prob > 0.3:
        print(f"    REVISAO MANUAL - Risco medio")
        print(f"    Acao: Encaminhar para analista")
        return "REVIEW"

    else:
        print(f"    APROVADO - Transacao segura")
        return "APPROVED"


def run_simulation(n_transactions: int = 10):
    """
    Simula multiplas transacoes.

    Args:
        n_transactions: Numero de transacoes a simular
    """
    print("\n" + "="*60)
    print("     SIMULADOR DE GATEWAY DE PAGAMENTOS")
    print("     Demonstracao de integracao com API de Fraude")
    print("="*60)

    # Verificar se API esta rodando
    try:
        health = requests.get(f"{API_URL}/health")
        if health.status_code != 200:
            raise Exception("API unhealthy")
        print(f"\n[OK] API conectada: {API_URL}")
    except:
        print(f"\n[ERRO] API nao encontrada em {API_URL}")
        print("Execute primeiro: make api")
        return

    # Simular transacoes
    customers = ["Maria Silva", "Joao Santos", "Ana Costa", "Pedro Lima", "Julia Souza"]

    stats = {"APPROVED": 0, "BLOCKED": 0, "REVIEW": 0}

    for i in range(n_transactions):
        customer = random.choice(customers)
        transaction = simulate_transaction()

        result = process_payment(
            customer=customer,
            amount=transaction["amount"],
            transaction=transaction
        )

        stats[result] += 1
        time.sleep(0.5)  # Pausa entre transacoes

    # Resumo
    print("\n" + "="*60)
    print("RESUMO DA SIMULACAO")
    print("="*60)
    print(f"Total de transacoes: {n_transactions}")
    print(f"Aprovadas: {stats['APPROVED']} ({100*stats['APPROVED']/n_transactions:.0f}%)")
    print(f"Bloqueadas: {stats['BLOCKED']} ({100*stats['BLOCKED']/n_transactions:.0f}%)")
    print(f"Em revisao: {stats['REVIEW']} ({100*stats['REVIEW']/n_transactions:.0f}%)")


def interactive_mode():
    """Modo interativo para testar transacoes manualmente."""
    print("\n" + "="*60)
    print("     MODO INTERATIVO")
    print("="*60)
    print("Digite os dados da transacao (ou 'q' para sair)")

    while True:
        print("\n" + "-"*40)

        try:
            amount = input("Valor (R$): ")
            if amount.lower() == 'q':
                break
            amount = float(amount)

            # Usar valores default para V features
            transaction = {
                "amount": amount,
                "v14": float(input("V14 (ou Enter para 0): ") or 0),
            }

            result = check_fraud(transaction)

            print(f"\n--- RESULTADO ---")
            print(f"Fraude: {'SIM' if result['is_fraud'] else 'NAO'}")
            print(f"Probabilidade: {result['fraud_probability']:.1%}")
            print(f"Risco: {result['risk_level'].upper()}")

        except ValueError:
            print("Valor invalido!")
        except requests.exceptions.ConnectionError:
            print("[ERRO] API nao conectada. Execute: make api")
            break


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "-i":
        interactive_mode()
    else:
        run_simulation(10)
