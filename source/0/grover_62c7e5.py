# https://github.com/yutakahirano/qc/blob/130abefe28c920f767e82bb9408e33c58766df17/simulation/grover.py
# cf. https://qiskit.org/textbook/ja/ch-algorithms/grover.html


import math
import qiskit
from qiskit import QuantumCircuit, Aer, transpile


def run(qc):
    sim = Aer.get_backend('aer_simulator')
    copy = qc.copy()
    copy.save_statevector()
    return qiskit.execute(copy, sim).result()


def diffuser(n):
    circuit = QuantumCircuit(n)
    circuit.h(range(n))
    circuit.x(range(n))

    circuit.h(n - 1)
    circuit.mcx(list(range(n - 1)), n - 1)
    circuit.h(n - 1)

    circuit.x(range(n))
    circuit.h(range(n))

    gate = circuit.to_gate()
    gate.name = 'diffuser({})'.format(n)
    return gate


def grover(n, uf, k = None):
    if k is None:
        k = math.floor(math.sqrt(2**n) * math.pi / 4 - 0.5)
    qc = QuantumCircuit(n)

    qc.h(range(n))
    for _ in range(k):
        qc.append(uf, range(n))
        qc.append(diffuser(n), range(n))

    return qc


def example(n, k = None):
    # Construct an example U_f. In this example, |1....1> is the solution.
    ufc = QuantumCircuit(n)
    ufc.h(n - 1)
    ufc.mcx(list(range(n - 1)), n - 1)
    ufc.h(n - 1)
    uf = ufc.to_gate()
    uf.name = 'Uf({})'.format(n)

    return run(grover(n, uf, k))
