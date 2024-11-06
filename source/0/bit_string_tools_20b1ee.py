# https://github.com/jean-philippe-arias-zapata/Qiskit-Toolbox/blob/541e096baccbc5435d4e85f345e68a8a6f94829c/BitStringTools/Bit_string_tools.py
from qiskit import QuantumRegister
from qiskit.extensions.standard.x import XGate
from AbstractGates.qiwiGate import qiwiGate


def to_binary(decimal_number, num_qubits, least_significant_bit_first=True):
    to_bin = format(decimal_number, 'b')
    if len(to_bin) != num_qubits:
        difference = num_qubits - len(to_bin)
        while(difference != 0):
            to_bin = '0' + to_bin 
            difference = difference - 1
    to_bin = to_bin[::-1] 
    if least_significant_bit_first == False:
        to_bin = to_bin[::-1]    
    return to_bin


def to_decimal(binary_string, least_significant_bit_first=True):
    number = 0 
    n = len(binary_string)
    if least_significant_bit_first == False:
            binary_string = binary_string[::-1]
    for i in range(n):
            number = number + int(binary_string[i]) * 2 ** i
    return number


def x_gates_region(circuit, qubits, number, least_significant_bit_first=True):
    n_qubits = len(qubits)
    string = to_binary(number, n_qubits, least_significant_bit_first)
    for i in range(n_qubits):
        if string[i] == '0':
              circuit.x(qubits[n_qubits - i - 1])
    return circuit 


class XRegionGate(qiwiGate):
    """X-Region gate. """
    
    def __init__(self, num_qubits, number, least_significant_bit_first=True):
        self.num_qubits = num_qubits
        self.least_significant_bit_first = least_significant_bit_first
        self.number = number
        super().__init__(name=f"XRegion(" + str(number) + ")", num_qubits=num_qubits, params=[], least_significant_bit_first=least_significant_bit_first)
    
    def _define(self):
        definition = []
        q = QuantumRegister(self.num_qubits)
        binary = to_binary(self.number, self.num_qubits, self.least_significant_bit_first)
        for i in range(self.num_qubits):
            if binary[i] == '0':
                definition.append((XGate(), [q[self.num_qubits - i - 1]], []))
        self.definition = definition
