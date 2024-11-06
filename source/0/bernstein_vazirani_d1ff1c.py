# https://github.com/Crabster/qiskit-learning/blob/3f14c39ee294f42e3f83a588910b659280556a68/circuits/bernstein_vazirani.py
import qiskit
from .common_gates import *
import random


def bernstein_vazirani_oracle(n, f_str):
    qc = qiskit.QuantumCircuit(n + 1)

    qc.x(n)
    qc.h(n)

    for i, bit in enumerate(f_str):
        if bit == '1':
            qc.cx(n - i - 1, n)

    qc.h(n)
    qc.x(n)

    print(qc.draw())
    gate = qc.to_gate()
    gate.name = "BV_O"
    return gate


def bernstein_vazirani_circuit(n, bv_oracle):
    qc = qiskit.QuantumCircuit(n + 1)

    for i in range(n):
        qc.h(i)

    qc.barrier()
    qc.append(bv_oracle, range(n + 1))
    qc.barrier()
    
    for i in range(n):
        qc.h(i)

    return qc


def bernstein_vazirani_example():
    n = 2
    f_str = random.choice(["00", "01", "10", "11"])
    print(f"The function string is {f_str}")

    qc = qiskit.QuantumCircuit(n + 1, n)

    bv_oracle = bernstein_vazirani_oracle(n, f_str)
    bv_qc = bernstein_vazirani_circuit(n, bv_oracle)
    qc.append(bv_qc, range(n + 1))

    qc.measure(range(n), range(n))
    print(qc.draw())
    return qc
