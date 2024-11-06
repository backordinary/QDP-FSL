# https://github.com/jean-philippe-arias-zapata/Qiskit-Toolbox/blob/541e096baccbc5435d4e85f345e68a8a6f94829c/AbstractGates/ToQiwiGate.py
from qiskit import QuantumRegister
from qiwiGate import qiwiGate


class ToQiwiGate(qiwiGate):
    """Conversion to a qiwiGate object."""
    
    def __init__(self, dummy_gate, least_significant_bit_first=True):
        self.num_qubits = dummy_gate.num_qubits
        self.least_significant_bit_first = least_significant_bit_first
        self.dummy_gate = dummy_gate
        super().__init__(name=f"Benji gate", num_qubits=dummy_gate.num_qubits, params=[], least_significant_bit_first=least_significant_bit_first)
        
    def _define(self):
        definition = []
        q = QuantumRegister(self.num_qubits)
        if isinstance(self.dummy_gate, qiwiGate) == True:
            if self.dummy_gate.least_significant_bit_first != self.least_significant_bit_first:
                raise NameError("The desired convention and the " + self.dummy_gate.name + " convention are not compatible.")  
        if isinstance(self.dummygate, qiwiGate) == False and self.least_significant_bit_first == False:
            q = q[::-1]
        definition.append((self.dummy_gate, q, []))
        if isinstance(self.dummygate, qiwiGate) == False and self.least_significant_bit_first == False:
            q = q[::-1]
        self.definition = definition