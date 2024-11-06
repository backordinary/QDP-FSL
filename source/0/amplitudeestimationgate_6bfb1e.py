# https://github.com/jean-philippe-arias-zapata/Qiskit-Toolbox/blob/541e096baccbc5435d4e85f345e68a8a6f94829c/AbstractGates/AmplitudeEstimationGate.py
from qiskit import QuantumRegister
from AbstractGates.qiwiGate import qiwiGate
from AbstractGates.ControlGate import ControlGate
#from AbstractGates.ControlGate import LambdaGate
from BinaryOracles.Boolean_preparation import to_list
from Grover.GroverGate import GroverGate
from QFT.QFTGate import QFTGate


#TO BE TESTED


class AmplitudeEstimationGate(qiwiGate):
    """Amplitude Estimation gate."""
    
    def __init__(self, num_accuracy_qubits, list_values, A_gate, least_significant_bit_first=True):
        self.num_qubits = num_accuracy_qubits + A_gate.num_qubits
        self.least_significant_bit_first = least_significant_bit_first
        self.num_accuracy_qubits = num_accuracy_qubits
        self.list_values = to_list(list_values)
        self.A_gate = A_gate
        super().__init__(name=f"Amplitude Estimation(" + A_gate.name +")", num_qubits=num_accuracy_qubits + A_gate.num_qubits, params=[], least_significant_bit_first=least_significant_bit_first)
        
    def _define(self):
        definition = []
        q = QuantumRegister(self.num_qubits)
        accuracy_qubits = q[:self.num_accuracy_qubits]
        definition.append((QFTGate(self.num_accuracy_qubits, False, self.least_significant_bit_first), accuracy_qubits, []))
        for i in range(2 ** self.num_accuracy_qubits):
            j = 1
            while j <= i:
                definition.append((ControlGate(self.num_accuracy_qubits, i, GroverGate(self.list_values, self.A_gate, not(self.least_significant_bit_first)), not(self.least_significant_bit_first)), q, []))
        #definition.append((LambdaGate(self.num_accuracy_qubits, GroverGate(self.list_values, self.A_gate, not(self.least_significant_bit_first)), not(self.least_significant_bit_first)), q, []))
        definition.append((QFTGate(self.num_accuracy_qubits, False, self.least_significant_bit_first).inverse(), accuracy_qubits, []))
        self.definition = definition
        
        