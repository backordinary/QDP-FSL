# https://github.com/HenningBuhl/QuantumComputing/blob/a9d64890d24c6ba24685c5b4fc6a608bfed9848b/Simons/Simons.py
# importing Qiskit
from qiskit import QuantumCircuit, transpile, assemble

from QuantumAlgorithm import QuantumAlgorithm
from scipy.linalg import lu
import numpy as np


class Simons(QuantumAlgorithm):
    def __init__(self, args):
        super().__init__("Simons", args)

    def get_circuit(self):
        b = self.args.simons_b

        n = len(b)
        circuit = QuantumCircuit(n * 2, n)

        # Apply Hadamard gates before querying the oracle
        circuit.h(range(n))

        # Apply barrier for visual separation
        circuit.barrier()

        # Oracle.
        oracle = self.get_oracle(b)
        oracle = oracle.to_gate()
        oracle.name = 'oracle'
        circuit.append(oracle, range(n*2), [])

        # Apply barrier for visual separation
        circuit.barrier()

        # Apply Hadamard gates to the input register
        circuit.h(range(n))

        # Measure qubits
        circuit.measure(range(n), range(n))

        return circuit

    def get_oracle(self, b):
        b = b[::-1]  # reverse b for easy iteration
        n = len(b)
        qc = QuantumCircuit(n * 2)
        # Do copy; |x>|0> -> |x>|x>
        for q in range(n):
            qc.cx(q, q + n)
        if '1' not in b:
            return qc  # 1:1 mapping, so just exit
        i = b.find('1')  # index of first non-zero bit in b
        # Do |x> -> |s.x> on condition that q_i is 1
        for q in range(n):
            if b[q] == '1':
                qc.cx(i, (q) + n)
        return qc

    def get_b_from_counts(self, _counts):  # TODO Only works for some bs... and randomly fails...
        counts = []
        for c in _counts:
            counts.append(c)

        list = []
        for c in counts:
            sub_list = []
            for i in c:
                sub_list.append(i)
            list.append(sub_list)
        a = np.array(list)

        _, _, u = lu(a)
        print(u)

        b = ''
        for i in u[0]:
            b += str(int(i))
        return b
