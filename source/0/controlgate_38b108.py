# https://github.com/jean-philippe-arias-zapata/Qiskit-Toolbox/blob/541e096baccbc5435d4e85f345e68a8a6f94829c/AbstractGates/ControlGate.py
from qiskit import QuantumRegister
from AbstractGates.qiwiGate import qiwiGate
from BitStringTools.Bit_string_tools import XRegionGate


class ControlGate(qiwiGate):
    """Control gate."""
    
    def __init__(self, num_ctrl_qubits, number, gate, least_significant_bit_first=True):
        self.num_qubits = num_ctrl_qubits + gate.num_qubits
        self.least_significant_bit_first = least_significant_bit_first
        self.num_ctrl_qubits = num_ctrl_qubits
        self.number = number
        self.gate = gate
        super().__init__(name=f"Control-" + gate.name + "(" + str(number) + ")", num_qubits=num_ctrl_qubits + gate.num_qubits, params=[], least_significant_bit_first=least_significant_bit_first)
        
        
    def _define(self):
        if isinstance(self.gate, qiwiGate) == True:
            if self.least_significant_bit_first != self.gate.least_significant_bit_first:
                raise NameError("The desired convention and the " + self.gate.name + " convention are not compatible.")                  
        if self.number >= 0 and self.number < 2 ** self.num_ctrl_qubits:
            definition = []
            q = QuantumRegister(self.num_qubits)
            ctrl_q = q[:self.num_ctrl_qubits]
            target_q = q[self.num_ctrl_qubits:]
            definition.append((XRegionGate(self.num_ctrl_qubits, self.number, self.least_significant_bit_first), ctrl_q, []))
            if isinstance(self.gate, qiwiGate) == False and self.least_significant_bit_first == False:
                q = ctrl_q + target_q[::-1]    
            definition.append((self.gate.control(self.num_ctrl_qubits), q, []))
            if isinstance(self.gate, qiwiGate) == False and self.least_significant_bit_first == False:
                q = ctrl_q + target_q[::-1]    
            definition.append((XRegionGate(self.num_ctrl_qubits, self.number, self.least_significant_bit_first), ctrl_q, []))
            self.definition = definition
        else:
            raise NameError("The chosen number is not compatible with the number of control qubits.")
            
            
""" To be finished. Need Benjamin's help.


class LambdaGate(qiwiGate):
    Lambda gate (found in arXiv.0005055).
    
    def __init__(self, num_ctrl_qubits, gate, least_significant_bit_first=True):
        self.num_qubits = num_ctrl_qubits + gate.num_qubits
        self.least_significant_bit_first = least_significant_bit_first
        self.num_ctrl_qubits = num_ctrl_qubits
        self.gate = gate
        super().__init__(name=f"Lambda(" + gate.name + ")", num_qubits=num_ctrl_qubits + gate.num_qubits, params=[], least_significant_bit_first=least_significant_bit_first)
        
    def _define(self):"""
        