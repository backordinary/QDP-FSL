# https://github.com/Crabster/qiskit-learning/blob/3f14c39ee294f42e3f83a588910b659280556a68/circuits/quantum_phase_estimation.py
import qiskit

from .common_gates import *
from .quantum_fourier_transform import qft_gate
from math import pi

import random

def qpe_gate(n, c_u):
    qc = qiskit.QuantumCircuit(n + 1)

    qc.x(n)

    for i in range(n): 
        qc.h(i)
        for _ in range(2**i):
            qc.append(c_u, [i, n])


    qc.append(qft_gate(n).inverse(), range(n))

    gate = qc.to_gate()
    gate.name = "QPE"
    return gate

def qpe_example():
    n = 3
    c_u = qiskit.QuantumCircuit(2)
    c_u.cp(pi / 4, 0, 1)
    c_u = c_u.to_gate()
    c_u.name = "C_U"

    qc = qiskit.QuantumCircuit(n + 1, n)

    qpe = qpe_gate(n, c_u)

    qc.append(qpe, range(n + 1))

    qc.measure(range(n), range(n))

    return qc
