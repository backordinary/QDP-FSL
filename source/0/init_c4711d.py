# https://github.com/PankidT/qiskit-textbook/blob/cbb80ad722af18f722fc14b04c184ef02f802b61/qiskit-textbook-src/qiskit_textbook/problems/__init__.py
from qiskit import QuantumCircuit
import numpy as np
from qiskit.circuit.library import Diagonal

def dj_problem_oracle(problem, to_gate=True):
    """Returns a 5-qubit Deutsch-Joza Oracle"""
    qc = QuantumCircuit(5)
    int(problem)
    if problem == 1:
        qc.cx(0, 4)
        qc.cx(1, 4)
        qc.cx(2, 4)
        qc.cx(3, 4)
    elif problem == 2:
        qc.cx(0, 4)
        qc.cx(0, 4)
    elif problem == 3:
        qc.ccx(0,2,4)
        qc.ccx(1,2,4)
        qc.x(0)
        qc.x(1)
        qc.x(2)
        qc.ccx(0,2,4)
        qc.ccx(1,3,4)
    elif problem == 4:
        qc.cx(2,4)
    else:
        print("There are only currently 4 oracles in this problem set, returning empty (balanced) gate")
    if to_gate:
        return qc.to_gate()
    else:
        return qc


def grover_problem_oracle(n, variant=0, print_solutions=False):
    np.random.seed(variant)
    if n < 3:
        nsolutions = 1
    else:
        nsolutions = np.random.randint(1, np.ceil((2**n)/4))
    diagonal_elements = [-1]*nsolutions + [1]*((2**n) - nsolutions)
    np.random.shuffle(diagonal_elements)
    oracle_gate = Diagonal(diagonal_elements)
    oracle_gate.name = "Oracle\nn=%i, var=%i" % (n, variant)
    if print_solutions:
        print("Solutions:")
        for idx, e in enumerate(diagonal_elements):
            if e < 1:
                print("|%s>" % format(idx, "0%ib" % n))
    return oracle_gate
