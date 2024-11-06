# https://github.com/sami-b95/Qiskit-Toolbox/blob/14a378f4bb3594a441ae0a2f7e0af59dd4c07adf/BinaryOracles/Test/Grover.py
from qiskit import QuantumCircuit
from qiskit.aqua.circuits.gates import mcry
import numpy as np


### UNARY LOWER THAN


def strictly_lower_than_binary_oracle(circuit, ancillae_qubits, threshold_qubits, threshold):
    threshold = int(threshold)
    main_ancilla_qubit = ancillae_qubits[0]
    n_qubits = len(threshold_qubits)     
    if len(ancillae_qubits) == 1:
        if threshold > n_qubits - 1:
            circuit.x(main_ancilla_qubit)
        elif threshold <= 0:
                raise NameError('This threshold cannot be studied because of lack of ancillae qubits.')
        else:
            circuit.x(threshold_qubits[threshold:])
            circuit.mct(threshold_qubits[threshold:], main_ancilla_qubit, threshold_qubits[0:threshold])
            circuit.x(threshold_qubits[threshold:])
    else:
        if threshold > n_qubits - 1:
            circuit.x(main_ancilla_qubit)
        else:
            circuit.x(threshold_qubits[max(0, threshold):])
            circuit.mct(threshold_qubits[max(0, threshold):], main_ancilla_qubit, ancillae_qubits[1:])
            circuit.x(threshold_qubits[max(0, threshold):])
    return circuit


def reflection_strictly_lower_than(circuit, ancillae_qubits, threshold_qubits, threshold):
    main_ancilla_qubit = ancillae_qubits[0]
    circuit.h(main_ancilla_qubit)
    circuit.x(main_ancilla_qubit)
    circuit.h(main_ancilla_qubit)
    circuit = strictly_lower_than_binary_oracle(circuit, ancillae_qubits, threshold_qubits, threshold)
    circuit.h(main_ancilla_qubit)
    circuit.x(main_ancilla_qubit)
    circuit.h(main_ancilla_qubit)
    circuit = strictly_lower_than_binary_oracle(circuit, ancillae_qubits, threshold_qubits, threshold)
    return circuit


### CONTROLLED VERSIONS
    

def controlled_strictly_lower_than_binary_oracle(circuit, ancillae_qubits, control_qubits, threshold_qubits, threshold):
    threshold = int(threshold)
    second_ancilla_qubit = ancillae_qubits[1]
    n_qubits = len(threshold_qubits)     
    if len(ancillae_qubits) == 1:
        raise NameError('This threshold cannot be studied because of lack of ancillae qubits.')
    else:
        if threshold > n_qubits - 1:
            circuit.mct(control_qubits, second_ancilla_qubit, ancillae_qubits[2:])
        else:
            for qubit in threshold_qubits[max(0, threshold):]:
                circuit.mct(control_qubits, qubit, ancillae_qubits[2:])
            circuit.mct(list(control_qubits) + threshold_qubits[max(0, threshold):], second_ancilla_qubit, ancillae_qubits[2:])
            for qubit in threshold_qubits[max(0, threshold):]:
                circuit.mct(control_qubits, qubit, ancillae_qubits[2:])
    return circuit


def controlled_reflection_strictly_lower_than(circuit, ancillae_qubits, control_qubits, threshold_qubits, threshold):
    second_ancilla_qubit = ancillae_qubits[1]
    circuit = controlled_strictly_lower_than_binary_oracle(circuit, ancillae_qubits, control_qubits, threshold_qubits, threshold)
    circuit.mcry(-np.pi/2, control_qubits, second_ancilla_qubit, ancillae_qubits[2:], mode='basic')
    circuit.mct(control_qubits, second_ancilla_qubit, ancillae_qubits[2:])
    circuit.mcry(-np.pi/2, control_qubits, second_ancilla_qubit, ancillae_qubits[2:], mode='basic')
    circuit = controlled_strictly_lower_than_binary_oracle(circuit, ancillae_qubits, control_qubits, threshold_qubits, threshold)
    circuit.mcry(-np.pi/2, control_qubits, second_ancilla_qubit, ancillae_qubits[2:], mode='basic')
    circuit.mct(control_qubits, second_ancilla_qubit, ancillae_qubits[2:])
    circuit.mcry(-np.pi/2, control_qubits, second_ancilla_qubit, ancillae_qubits[2:], mode='basic')
    return circuit