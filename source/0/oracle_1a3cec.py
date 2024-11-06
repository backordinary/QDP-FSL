# https://github.com/sofferor/Quantum-Algorithms/blob/ee6678001d7d09e8edbd79c06b356dc12006c173/Deutsch-Jozsa%20Algorithm/oracle.py
import numpy as np
from qiskit import QuantumCircuit
from utils import Case


def oracle(case, n):
    oracle_qc = QuantumCircuit(n + 1)

    if case == Case.balanced:
        b = np.random.randint(1, 2 ** n)
        b_str = "{0:b}".format(b)
        for qubit in range(len(b_str)):
            if b_str[qubit] == '1':
                oracle_qc.x(qubit)

        for qubit in range(n):
            oracle_qc.cx(qubit, n)

        for qubit in range(len(b_str)):
            if b_str[qubit] == '1':
                oracle_qc.x(qubit)

    if case == Case.constant:
        output = np.random.randint(2)
        if output == 1:
            oracle_qc.x(n)

    print("Oracle Circuit - {case} case:\n{circuit}\n\n".format(case=case.name, circuit=oracle_qc.draw()))
    oracle_gate = oracle_qc.to_gate()
    oracle_gate.name = "Oracle"
    return oracle_gate
