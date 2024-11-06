# https://github.com/JanusQ/AlgorithmBenchmark/blob/b729a8986d99bfe838b74ba34f213bfa4bb5e115/dataset/dataset1/w_state.py
from qiskit import QuantumCircuit
import math


def F_gate(qc, i, j, n, k):
    theta = math.acos(math.sqrt(1. / (n - k + 1)))
    qc.ry(-theta, j)
    qc.cz(i, j)
    qc.ry(theta, j)


def get_cir(n_qubits):
    qc = QuantumCircuit(n_qubits, n_qubits)
    n = n_qubits

    qc.x(n - 1)

    for i in range(0, n - 1):
        F_gate(qc, n - 1 - i, n - 2 - i, n, i + 1)

    for i in range(0, n - 1):
        qc.cx(n - 2 - i, n - 1 - i)

    return qc
