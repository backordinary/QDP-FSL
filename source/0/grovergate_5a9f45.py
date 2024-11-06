# https://github.com/jean-philippe-arias-zapata/Qiskit-Toolbox/blob/541e096baccbc5435d4e85f345e68a8a6f94829c/Grover/GroverGate.py
from qiskit import QuantumRegister
from qiskit.extensions.standard.h import HGate
from AbstractGates.qiwiGate import qiwiGate
from BinaryOracles.Boolean_preparation import to_list
from BinaryOracles.ReflectionGate import ReflectionGate


class GroverAlgorithmGate(qiwiGate):
    """Grover algorithm gate."""
    
    def __init__(self, boolean_gate, least_significant_bit_first=True):
        self.num_qubits = boolean_gate.num_qubits
        self.least_significant_bit_first = least_significant_bit_first
        self.boolean_gate = boolean_gate
        super().__init__(name=f"Grover("+ boolean_gate.name +")", num_qubits=boolean_gate.num_qubits, params=[], least_significant_bit_first=least_significant_bit_first)
        
    def _define(self):
        if isinstance(self.boolean_gate, qiwiGate) == True:
            if self.least_significant_bit_first != self.boolean_gate.least_significant_bit_first:
                raise NameError("The desired convention and the " + self.boolean_gate.name + " convention are not compatible.")
        definition = []
        q = QuantumRegister(self.num_qubits)
        definition.append((self.boolean_gate, q, []))  
        for i in range(self.num_qubits):
            definition.append((HGate(), [q[i]], []))
        definition.append((ReflectionGate(self.num_qubits, 0, self.least_significant_bit_first), q, []))     
        for i in range(self.num_qubits):
            definition.append((HGate(), [q[i]], []))
        self.definition = definition
    
        
class GroverGate(qiwiGate):
    """Grover gate."""
    
    def __init__(self, list_values, A_gate, least_significant_bit_first=True):
        self.num_qubits = A_gate.num_qubits
        self.least_significant_bit_first = least_significant_bit_first
        self.list_values = to_list(list_values)
        self.A_gate = A_gate
        super().__init__(name=f"Grover("+ A_gate.name +")", num_qubits=A_gate.num_qubits, params=[], least_significant_bit_first=least_significant_bit_first)
        
    def _define(self):
        if isinstance(self.A_gate, qiwiGate) == True:
            if self.least_significant_bit_first != self.A_gate.least_significant_bit_first:
                raise NameError("The desired convention and the " + self.A_gate.name + " convention are not compatible.")
        definition = []
        q = QuantumRegister(self.num_qubits)
        definition.append((ReflectionGate(self.num_qubits, self.list_values, self.least_significant_bit_first), q, []))  
        definition.append((self.A_gate.inverse(), q, []))
        definition.append((ReflectionGate(self.num_qubits, 0, self.least_significant_bit_first), q, []))     
        definition.append((self.A_gate, q, []))
        self.definition = definition