# https://github.com/jean-philippe-arias-zapata/Qiskit-Toolbox/blob/541e096baccbc5435d4e85f345e68a8a6f94829c/QFT/QFTGate.py
from qiskit.circuit import Gate
from qiskit.extensions.standard.x import CnotGate
from qiskit.extensions.standard.h import HGate
from qiskit.extensions.standard.u1 import Cu1Gate
from qiskit import QuantumRegister
from AbstractGates.qiwiGate import qiwiGate
from math import pi


class DoSwapsGate(Gate):
    """Do Swaps gate."""
    
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        super().__init__(name=f"Do Swaps", num_qubits=num_qubits, params=[])
        
    def _define(self):
        definition = []
        q = QuantumRegister(self.num_qubits)
        for i in range(self.num_qubits // 2):
            definition.append((CnotGate(), [q[i], q[self.num_qubits - i - 1]], []))
            definition.append((CnotGate(), [q[self.num_qubits - i - 1], q[i]], []))
            definition.append((CnotGate(), [q[i], q[self.num_qubits - i - 1]], []))
        self.definition = definition
        

class QFTGate(qiwiGate):
    """Quantum Fourier Transform gate."""
    
    def __init__(self, num_qubits, bool_swaps=True, least_significant_bit_first=True):
        self.num_qubits = num_qubits
        self.least_significant_bit_first = least_significant_bit_first
        self.bool_swaps = bool_swaps
        super().__init__(name=f"QFT", num_qubits=num_qubits, params=[], least_significant_bit_first=least_significant_bit_first)
        
    def _define(self):
        definition = []
        q = QuantumRegister(self.num_qubits)
        if self.least_significant_bit_first == False:
            q = q[::-1]
        for i in range(self.num_qubits):
            definition.append((HGate(), [q[i]], []))
            for distance in range(self.num_qubits - i - 1):
                distance = distance + 1
                definition.append((Cu1Gate(pi / 2**distance), [q[distance + i], q[i]], []))
        if self.least_significant_bit_first == False:
            q = q[::-1]
        if self.bool_swaps:
            definition.append((DoSwapsGate(self.num_qubits), q, []))
        self.definition = definition