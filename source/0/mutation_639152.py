# https://github.com/Talkal13/Quantum/blob/ccda55776da0a3f5bd212a8566f0a1e367061a6f/MTF/mutation.py
from qiskit import QuantumCircuit
from qiskit.circuit.random import random_circuit
from random import random, randint

class Mutant:
    def __init__(self, qc : QuantumCircuit):
        self.qc = qc.copy()
        self.qc = self.mutate(self.qc)

    def mutate(self, qc : QuantumCircuit):
        # Delete a gate
        if (random() < 0.75):
            qc.data.pop(randint(0, len(qc.data) - 1))
        if (random() < 0.25):
            circ = random_circuit(qc.num_qubits, 1, measure=False)
            qc.append(circ.to_instruction(), qc.qregs)
        return qc
