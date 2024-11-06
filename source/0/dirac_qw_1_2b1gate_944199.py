# https://github.com/jean-philippe-arias-zapata/Qiskit-Toolbox/blob/541e096baccbc5435d4e85f345e68a8a6f94829c/QuantumAddition/Test/Dirac_QW_1%2B1Gate.py
from qiskit import QuantumRegister
import os
os.chdir('..')
from QuantumIncrementorGate import QuantumIncrementorGate
os.chdir('Test')
from qiskit.circuit import Gate
from qiskit.extensions.standard.x import XGate
from qiskit.extensions.standard.rx import RXGate

class DiracQWShiftGate(Gate):
    """Dirac QW Shift gate."""
    
    def __init__(self, num_qubits, epsilon):
        self.num_qubits = num_qubits
        super().__init__(name=f"Quantum Incrementor({epsilon})", num_qubits=num_qubits, params=[epsilon])
        
    def _define(self):
        self.definition = []
        q = QuantumRegister(self.num_qubits)
        self.definition.append((QuantumIncrementorGate(self.num_qubits, - self.params[0]).control(), q, []))
        self.definition.append((XGate(), [q[0]], []))
        self.definition.append((QuantumIncrementorGate(self.num_qubits, self.params[0]).control(), q, []))
        self.definition.append((XGate(), [q[0]], []))
        

class DiracQWCoinGate(Gate):
    """Dirac QW Coin gate."""

    def __init__(self, num_qubits, epsilon, mass):
        self.num_qubits = num_qubits
        super().__init__(name=f"Quantum Incrementor({epsilon})", num_qubits=num_qubits, params=[epsilon, mass])
        
    def _define(self):
        self.definition = []
        q = QuantumRegister(self.num_qubits)
        self.definition.append((RXGate(2 * self.params[0] * self.params[1]), [q[0]], []))
        
        
class DiracQWGate(Gate):
    """Dirac QW gate."""
    
    def __init__(self, num_qubits, epsilon, mass):
        self.num_qubits = num_qubits
        super().__init__(name=f"Quantum Incrementor({epsilon})", num_qubits=num_qubits, params=[epsilon, mass])
        
    def _define(self):
        self.definition = []
        q = QuantumRegister(self.num_qubits)
        self.definition.append((DiracQWCoinGate(self.num_qubits, self.params[0], self.params[1]), q, []))
        self.definition.append((DiracQWShiftGate(self.num_qubits, self.params[0]), q, []))