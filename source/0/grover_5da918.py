# https://github.com/HenningBuhl/QuantumComputing/blob/a9d64890d24c6ba24685c5b4fc6a608bfed9848b/Grover/grover.py
import matplotlib.pyplot as plt
import numpy as np

# importing Qiskit
from qiskit import IBMQ, Aer, assemble, transpile
import qiskit as q
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.circuit.library import XOR
from qiskit.providers.ibmq import least_busy

# import basic plot tools
from qiskit.visualization import plot_histogram

from QuantumAlgorithm import QuantumAlgorithm


class Grover(QuantumAlgorithm):
    """default 2 Qubits"""
    qubits = 3

    def __init__(self, args):
        super().__init__("Grover Algorithm", args)

    def get_circuit(self):
        qc = q.QuantumCircuit(self.args.grovers_n)
        qc.cz(0, 2)
        qc.cz(1, 2)

        grover_circuit = QuantumCircuit(self.args.grovers_n)
        grover_circuit = self.initialize_s(grover_circuit, [0, 1, 2])
        grover_circuit.append(self.oracle_ex3(), [0, 1, 2])
        grover_circuit.append(self.diffuser(), [0, 1, 2])

        return grover_circuit

    def diffuser(self):
        qc = QuantumCircuit(self.qubits)
        # Apply transformation |s> -> |00..0> (H-gates)
        for qubit in range(self.qubits):
            qc.h(qubit)
        # Apply transformation |00..0> -> |11..1> (X-gates)
        for qubit in range(self.qubits):
            qc.x(qubit)
        # Do multi-controlled-Z gate
        qc.h(self.qubits - 1)
        qc.mct(list(range(self.qubits - 1)), self.qubits - 1)  # multi-controlled-toffoli
        qc.h(self.qubits - 1)
        # Apply transformation |11..1> -> |00..0>
        for qubit in range(self.qubits):
            qc.x(qubit)
        # Apply transformation |00..0> -> |s>
        for qubit in range(self.qubits):
            qc.h(qubit)
        # We will return the diffuser as a gate
        # return qc
        U_s = qc.to_gate()
        U_s.name = "U$_s$"
        return U_s

    def oracle_ex3(self):
        qc = QuantumCircuit(3)
        qc.cz(0, 2)
        qc.cz(1, 2)
        oracle_ex3 = qc.to_gate()
        oracle_ex3.name = "U$_\omega$"
        return oracle_ex3

    def initialize_s(self, qc, qubits):
        """Apply a H-gate to 'qubits' in qc"""
        for q in qubits:
            qc.h(q)
        return qc
