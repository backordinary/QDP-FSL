# https://github.com/yahja-dev/QuantumLabs/blob/bcde084ed481e333fe5d159b6413bda4e61ead05/GroversAlgorithm/methods.py
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
import numpy as np


def oracle(nqubits, to_mark):
    """
    :param nqubits: number of qubits
    :param to_mark: list of indices to be marked with a phase of -1
    :return: oracle_circuit
    """
    oracle_circuit = QuantumCircuit(nqubits, name="Oracle")
    matrix = np.identity(2**nqubits)
    for index in to_mark:
        matrix[index][index] = -1

    oracle_circuit.unitary(Operator(matrix), range(nqubits))
    oracle_circuit.draw(output='mpl', filename='illustrations/oracle.png')
    return oracle_circuit


def diffuser(nqubits):
    """
    :param nqubits: number of qubits
    :return: diffuser_circuit
    """
    diffuser_circuit = QuantumCircuit(nqubits, name='Diffuser')
    diffuser_circuit.h(range(nqubits))
    diffuser_circuit.append(oracle(nqubits, [0]), range(nqubits))
    diffuser_circuit.h(range(nqubits))
    diffuser_circuit.draw(output='mpl', filename='illustrations/diffuser.png')

    return diffuser_circuit