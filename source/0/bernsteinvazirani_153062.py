# https://github.com/HenningBuhl/QuantumComputing/blob/a9d64890d24c6ba24685c5b4fc6a608bfed9848b/BernsteinVazirani/BernsteinVazirani.py
from QuantumAlgorithm import QuantumAlgorithm
import qiskit as q
import numpy as np


class BernsteinVazirani(QuantumAlgorithm):
    def __init__(self, args):
        super().__init__("Bernstein Vazirani", args)

    def get_circuit(self):
        s = self.args.bernstein_vazirani_s
        n = len(s)
        circuit = q.QuantumCircuit(n + 1, n)
        #s += 'b'

        # Put auxiliary in state |->
        circuit.h(n)
        circuit.z(n)

        # Apply Hadamard gates before querying the oracle
        for i in range(n):
            circuit.h(i)

        # Apply barrier
        circuit.barrier()

        # Apply the inner-product oracle
        oracle = self.get_oracle(s)
        oracle = oracle.to_gate()
        oracle.name = 'oracle'
        circuit.append(oracle, range(n+1), [])

        # Apply barrier
        circuit.barrier()

        # Apply Hadamard gates after querying the oracle
        for i in range(n):
            circuit.h(i)

        # Measurement
        for i in range(n):
            circuit.measure(i, i)

        return circuit

    def get_oracle(self, s):
        n = len(s)
        qc = q.QuantumCircuit(n +1)
        s = s[::-1]  # reverse s to fit qiskit's qubit ordering
        for _q in range(n):
            if s[_q] == '0':
                qc.i(_q)
            else:
                qc.cx(_q, n)
        return qc
