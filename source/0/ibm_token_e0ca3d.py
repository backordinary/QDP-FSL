# https://github.com/AsymptoticBrain/cwq/blob/d14ed7f55ffbb9047baea27ea32d867ec7e9c7e0/getting_started/IBM_token.py
"""
    Setup IBM quantum experience account, this will allow
    the qiskit env to send real quantum computers at IBM.
    The token is found under My account on the webpage.
"""

from qiskit import IBMQ

IBMQ.save_account('MyToken')