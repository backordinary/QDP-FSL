# https://github.com/jean-philippe-arias-zapata/Qiskit-Toolbox/blob/541e096baccbc5435d4e85f345e68a8a6f94829c/QuantumAddition/QuantumIncrementorGate.py
from qiskit import QuantumRegister
from qiskit.extensions.standard.u1 import U1Gate
from AbstractGates.qiwiGate import qiwiGate
from QFT.QFTGate import QFTGate
from math import pi



class QuantumIncrementorGate(qiwiGate):
    """Quantum Incrementor gate."""
    
    def __init__(self, num_qubits, epsilon, least_significant_bit_first=True):
        self.num_qubits = num_qubits
        self.least_significant_bit_first = least_significant_bit_first
        self.epsilon = epsilon
        super().__init__(name=f"Quantum Incrementor({epsilon})", num_qubits=num_qubits, params=[], least_significant_bit_first=least_significant_bit_first)
        
    def _define(self):
        definition = []
        q = QuantumRegister(self.num_qubits)
        if self.least_significant_bit_first == False:
            q = q[::-1]
        definition.append((QFTGate(self.num_qubits, bool_swaps=False), q, []))
        for i in range(self.num_qubits):
            definition.append((U1Gate(float(pi * self.epsilon)/2**i), [q[self.num_qubits - i - 1]], []))
        definition.append((QFTGate(self.num_qubits, bool_swaps=False).inverse(), q, []))
        if self.least_significant_bit_first == False:
            q = q[::-1]
        self.definition = definition