# https://github.com/jean-philippe-arias-zapata/Qiskit-Toolbox/blob/541e096baccbc5435d4e85f345e68a8a6f94829c/BinaryOracles/ReflectionGate.py
from qiskit import QuantumRegister
from qiskit.extensions.standard.h import HGate
from qiskit.extensions.standard.x import XGate
from AbstractGates.qiwiGate import qiwiGate
from BinaryOracles.Boolean_preparation import to_list
from BinaryOracles.OracleGate import OracleGate


class ReflectionGate(qiwiGate):
    """Reflection gate.
    
    If the input is:
    - an int, it will be seen as the control register value;
    - a list, it will be seen as a list of control register values.
    
    """
    
    def __init__(self, num_qubits, list_values, least_significant_bit_first=True):
        self.num_qubits = num_qubits
        self.least_significant_bit_first = least_significant_bit_first
        self.list_values = to_list(list_values)
        super().__init__(name=f"Reflection gate(" + str(list_values) +")", num_qubits=num_qubits, params=[], least_significant_bit_first=least_significant_bit_first)

    def _define(self):
        definition = []
        q = QuantumRegister(self.num_qubits)
        definition.append((HGate(), [q[self.num_qubits - 1]], []))
        definition.append((XGate(), [q[self.num_qubits - 1]], []))
        definition.append((HGate(), [q[self.num_qubits - 1]], []))
        definition.append((OracleGate(self.num_qubits, self.list_values, self.least_significant_bit_first), q, []))
        definition.append((HGate(), [q[self.num_qubits - 1]], []))
        definition.append((XGate(), [q[self.num_qubits - 1]], []))
        definition.append((HGate(), [q[self.num_qubits - 1]], []))
        definition.append((OracleGate(self.num_qubits, self.list_values, self.least_significant_bit_first), q, []))
        self.definition = definition