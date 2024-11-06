# https://github.com/sami-b95/Qiskit-Toolbox/blob/14a378f4bb3594a441ae0a2f7e0af59dd4c07adf/QuantumAddition/QuantumIncrementorGate.py
from math import pi
import os
os.chdir('../QFT')
from QFTGate import QFTGate
from QFT import quantum_fourier_transform
os.chdir('../QuantumAddition')
from qiskit.circuit import Gate
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.extensions.standard.u1 import U1Gate


class QuantumIncrementorGate(Gate):
    """Quantum Incrementor gate."""
    
    def __init__(self, num_qubits, epsilon):
        self.num_qubits = num_qubits
        super().__init__(name=f"Quantum Incrementor({epsilon})", num_qubits=num_qubits, params=[epsilon])
        
    def _define(self):
        self.definition = []
        q = QuantumRegister(self.num_qubits)
        self.definition.append((QFTGate(self.num_qubits), q, []))
        for i in range(self.num_qubits):
            self.definition.append((U1Gate(float(pi * self.params[0])/2**i), [q[self.num_qubits - i - 1]], []))
        self.definition.append((QFTGate(self.num_qubits).inverse(), q, []))
