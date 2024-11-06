# https://github.com/Crabster/qiskit-learning/blob/3f14c39ee294f42e3f83a588910b659280556a68/circuits/deutsch_jozsa.py
import qiskit
from .common_gates import *
import random


def deutsch_jozsa_oracle(n, f_str):
    qc = qiskit.QuantumCircuit(n + 1)

    for i, bit in enumerate(f_str):
        if bit == '1':
            qc.x(n - i - 1)

    qc.x(n)
    qc.h(n)

    for i in range(n):
        qc.cx(i, n)

    qc.h(n)
    qc.x(n)

    for i, bit in enumerate(f_str):
        if bit == '1':
            qc.x(n - i - 1)


    print(qc.draw())
    gate = qc.to_gate()
    gate.name = "DJ_O"
    return gate


def deutsch_jozsa_circuit(n, dj_oracle):
    qc = qiskit.QuantumCircuit(n + 1)

    for i in range(n):
        qc.h(i)

    qc.barrier()
    qc.append(dj_oracle, range(n + 1))
    qc.barrier()
    
    for i in range(n):
        qc.h(i)

    return qc


def deutsch_jozsa_example():
    print(f"TODO: doesn't work")
    n = 2
    f_str = random.choice(["00", "01", "10", "11"])
    print(f"The function string is {f_str}")

    qc = qiskit.QuantumCircuit(n + 1, n)

    dj_oracle = deutsch_jozsa_oracle(n, f_str)
    dj_qc = deutsch_jozsa_circuit(n, dj_oracle)
    qc.append(dj_qc, range(n + 1))

    qc.measure(range(n), range(n))
    print(qc.draw())
    return qc
