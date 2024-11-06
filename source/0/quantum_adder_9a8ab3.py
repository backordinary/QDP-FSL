# https://github.com/jean-philippe-arias-zapata/Qiskit-Toolbox/blob/541e096baccbc5435d4e85f345e68a8a6f94829c/QuantumAddition/Quantum_adder.py
from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister, Aer, execute
import os
os.chdir('../QFT')
from QFT import quantum_fourier_transform, do_swaps
os.chdir('../QuantumAddition')
from math import pi


def quantum_adder(circuit, to_add_qubits, target_qubits, least_significant_bit_first=True):
    n_qubits = len(target_qubits)
    if n_qubits != len(to_add_qubits):
        raise NameError('The two quantum registers have not the same size.')
    else:
        if least_significant_bit_first == False:
            target_qubits = target_qubits[::-1]
            to_add_qubits = to_add_qubits[::-1]
        circuit = quantum_fourier_transform(circuit, target_qubits, inverse=False, bool_swaps=False)
        for i in range(n_qubits):
            for j in range(n_qubits - i): 
                circuit.cu1(pi / 2**j, to_add_qubits[n_qubits - i - 1], target_qubits[n_qubits - i - 1 - j])
        circuit = quantum_fourier_transform(circuit, target_qubits, inverse=True, bool_swaps=False)
        if least_significant_bit_first == False:
            target_qubits = target_qubits[::-1]
            to_add_qubits = to_add_qubits[::-1]
        return circuit