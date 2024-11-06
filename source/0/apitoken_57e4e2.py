# https://github.com/Scdk/shor/blob/b7dd48604b87d506358f26d9a46169fa048de97c/apitoken.py
# Proposta para isolar o token dos outros programas
# É necessário rodar apenas uma vez, ou quando não tiver conexão com a IBM
from qiskit import IBMQ

apitoken = 'TOKEN'

if __name__ == "__main__":
    IBMQ.save_account(apitoken)
