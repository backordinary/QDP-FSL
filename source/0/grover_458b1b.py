# https://github.com/emre6943/QuantumProject/blob/86daae688f726bf6b258d2e7e8ebcffa5c4192c6/grover.py
import os
import matplotlib.pyplot as plt
import numpy as np
import math
from getpass import getpass
from quantuminspire.credentials import load_account, get_token_authentication, get_basic_authentication

from qiskit.circuit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import execute

from quantuminspire.qiskit import QI

QI_EMAIL = os.getenv('QI_EMAIL')
QI_PASSWORD = os.getenv('QI_PASSWORD')
QI_URL = os.getenv('API_URL', 'https://api.quantum-inspire.com/')


# Just creates an array from 0 to num of qubits - 1
def all_bits(q_num):
    arr = []
    for x in range(q_num):
        arr.append(x)
    return arr


def get_authentication():
    """ Gets the authentication for connecting to the Quantum Inspire API."""
    token = load_account()
    if token is not None:
        return get_token_authentication(token)
    else:
        if QI_EMAIL is None or QI_PASSWORD is None:
            print('Enter email:')
            email = input()
            print('Enter password')
            password = getpass()
        else:
            email, password = QI_EMAIL, QI_PASSWORD
        return get_basic_authentication(email, password)


def initialize_s(qc, qubits):
    """Apply a H-gate to qubits in quantum circuit qc."""
    for q in qubits:
        qc.h(q)
    return qc


def oracle(answer):
    """
    Creates the oracle depending on the output wanted
    :param answer: the wanted element
    :return: quantum circuit
    """
    answer = answer[::-1]
    num = len(answer)
    qc = QuantumCircuit(num)

    zero = []
    one = []
    for i, val in enumerate(answer):
        if val:
            one.append(i)
        else:
            zero.append(i)

    # Placing Zeros
    for x in zero:
        qc.x(x)

    # Connectors
    if num == 2:
        qc.cz(0, 1)
    else:
        qc.h(num - 1)
        qc.mct(list(range(num - 1)), num - 1)  # multi-controlled-toffoli
        qc.h(num - 1)

    # Placing Zeros
    for x in zero:
        qc.x(x)

    gate = qc.to_gate()
    gate.name = "Oracle"
    print("Oracle")
    print(qc.draw())

    return gate


def diffuser(nqubits):
    """Creates a diffuser for a specific number of qubits."""
    # IBM tutorial helped
    qc = QuantumCircuit(nqubits)
    # H all
    for qubit in range(nqubits):
        qc.h(qubit)
    # Not all
    for qubit in range(nqubits):
        qc.x(qubit)
    # Do multi-controlled-Z gate
    qc.h(nqubits-1)
    qc.mct(list(range(nqubits-1)), nqubits-1)  # multi-controlled-toffoli
    qc.h(nqubits-1)
    # Not all
    for qubit in range(nqubits):
        qc.x(qubit)
    # H all
    for qubit in range(nqubits):
        qc.h(qubit)

    gate = qc.to_gate()
    gate.name = "Diffuser"
    print("The Diffuser")
    print(qc.draw())
    return gate


def plot_results(probabilities_histogram, size):
    """Plot the results with states on x-axis and their probabilities on y-axis."""
    names = []
    values = []

    for state, val in probabilities_histogram.items():
        names.append(state)
        values.append(val)
    plt.figure(figsize=(size[0], size[1]))
    plt.bar(names, values)
    plt.xlabel('Probabilities')
    plt.ylabel('States')
    plt.title('Results')
    plt.show()


def grover(qc, answer, bits):
    #this r adds the sqrtN complexity
    r = math.floor(np.pi / 4 * math.sqrt((2 ** len(bits)) / len(answer)))
    if r == 0:
        r = 1
    print(str(r) + " iterations made")
    for i in range(r):
        for an in answer:
            qc.append(oracle(an), bits)
        qc.append(diffuser(len(bits)), bits)
    return qc


def cheap_grover(n_bits, answers, qi_backend):
    """This algorithm checks whether the number of search elements isn't over the limit of what one circuit can find.
    If this is true, the circuit is split into two. """
    limit = 2 ** n_bits / 2
    if len(answers) >= limit:  # the circuit has to be split up
        answers = np.array_split(answers, 2)
        for an in answers:
            bits = all_bits(n_bits)

            q = QuantumRegister(n_bits)
            b = ClassicalRegister(n_bits)
            circuit = QuantumCircuit(q, b)

            # compexity is O(n)
            circuit = initialize_s(circuit, bits)

            circuit = grover(circuit, an, bits)

            # compexity is O(n)
            circuit.measure(q, b)

            print(circuit.draw())

            qi_job = execute(circuit, backend=qi_backend, shots=1024)
            qi_result = qi_job.result()
            probabilities_histogram = qi_result.get_probabilities(circuit)
            plot_results(probabilities_histogram, [120, 20])
    else:  # we can execute the algorithm with one circuit
        bits = all_bits(n_bits)

        q = QuantumRegister(n_bits)
        b = ClassicalRegister(n_bits)
        circuit = QuantumCircuit(q, b)

        # compexity is O(n)
        circuit = initialize_s(circuit, bits)

        circuit = grover(circuit, answers, bits)

        # compexity is O(n)
        circuit.measure(q, b)

        print(circuit.draw())

        qi_job = execute(circuit, backend=qi_backend, shots=1024)
        qi_result = qi_job.result()
        probabilities_histogram = qi_result.get_probabilities(circuit)
        plot_results(probabilities_histogram, [120, 20])


if __name__ == '__main__':

    authentication = get_authentication()
    QI.set_authentication(authentication, QI_URL)

    # this is where you choose the backend, the options are: Starmon-5, Spin-2, or QX single-node simulator
    qi_backend = QI.get_backend('QX single-node simulator')

    # An example for how to search for multiple elements
    cheap_grover(3, [[True, True, True], [True, True, False], [True, False, False], [False, False, False]], qi_backend)
