# https://github.com/Gcuencam/full-tomography/blob/331217f6a97038f0e2e74cd63a230c6aec1ea583/src/states/plus.py
from qiskit import QuantumCircuit
import numpy as np
import math
from qiskit.quantum_info import Statevector


def build(circuit, referencePosition, size):
    circuit.h(referencePosition)
    return circuit


def plus_state(size: int):
    circuit = QuantumCircuit(size, size)

    circuit.h(0)

    return circuit


def plus_state_rho():
    ket_plus_state = np.dot(1 / math.sqrt(2), np.array([[1, 1]]).T)
    bra_plus_state = np.dot(1 / math.sqrt(2), np.array([[1, 1]]))
    return np.dot(ket_plus_state, bra_plus_state)


def get_plus_state_vector(qc_size):
    circuit = QuantumCircuit(qc_size, qc_size)
    plus_qc = build(circuit, 0, qc_size)
    psi = Statevector.from_instruction(plus_qc)
    return np.array(psi)


if __name__ == '__main__':
    # Probabilities for measuring both qubits. In any measurement is applied in the circuit, this won't work.
    q_size = 1
    print(get_plus_state_vector(q_size))
