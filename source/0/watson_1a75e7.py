# https://github.com/emre6943/QuantumProject/blob/86daae688f726bf6b258d2e7e8ebcffa5c4192c6/watson.py
import os
import matplotlib.pyplot as plt
import numpy as np
import math
from getpass import getpass
from quantuminspire.credentials import load_account, get_token_authentication, get_basic_authentication

from qiskit.circuit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import execute

from quantuminspire.qiskit import QI

from grover import get_authentication, plot_results

QI_EMAIL = os.getenv('QI_EMAIL')
QI_PASSWORD = os.getenv('QI_PASSWORD')
QI_URL = os.getenv('API_URL', 'https://api.quantum-inspire.com/')


def oracle(search_element):
    """
    Creates the oracle depending on the search element.
    """
    qc = QuantumCircuit(2)

    # oracles
    if search_element == 0:  # 00
        qc.x(0)
        qc.x(1)
        qc.cz(0, 1)
        qc.x(0)
        qc.x(1)
    elif search_element == 1:  # 01
        qc.x(1)
        qc.cz(0, 1)
        qc.x(1)
    elif search_element == 2:  # 10
        qc.x(0)
        qc.cz(0, 1)
        qc.x(0)
    elif search_element == 3:  # 11
        qc.cz(0, 1)
    else:
        raise ValueError("Searching for non-existing element. The possibilities are: 0, 1, 2, 3.")

    gate = qc.to_gate()
    gate.name = "Oracle for state {}".format(search_element)

    return gate


def watson(qc, search_element):
    """Creates a circuit for the alternative implementation from Watson et al. (2018).

    Here, the answer is a state in decimal notation, so 0 for |00>, 1 for |01>, 2 for |10>, and 3 for |11>.
    """
    # initialise
    for i in [0, 1]:
        qc.ry(np.pi / 2, i)

    qc.append(oracle(search_element), [0, 1])

    # diffuser
    for i in [0, 1]:
        qc.ry(np.pi / 2, i)
    qc.append(oracle(0), [0, 1])
    for i in [0, 1]:
        qc.ry(np.pi / 2, i)

    return qc


if __name__ == '__main__':

    authentication = get_authentication()
    QI.set_authentication(authentication, QI_URL)

    # this is where you choose the backend
    qi_backend = QI.get_backend('QX single-node simulator')

    q = QuantumRegister(2)
    c = ClassicalRegister(2)
    qc = QuantumCircuit(q, c)

    qc = watson(qc, search_element=0)

    print(qc.draw())

    qi_job = execute(qc, backend=qi_backend, shots=1024)
    qi_result = qi_job.result()
    probabilities_histogram = qi_result.get_probabilities(qc)
    plot_results(probabilities_histogram, [120, 20])
