# https://github.com/physicalattraction/quantum-computing-tutorial/blob/bbeb8a86cb8ae06bd8072505d826a3eec6ce861a/src/qiskit_1_3_representing_qubit_states.py
"""
https://qiskit.org/textbook/ch-states/representing-qubit-states.html
"""

from math import sqrt
from qiskit import QuantumCircuit

from utils import draw_quantum_circuit, bit_to, State, normalize


def initialize():
    qc = QuantumCircuit(1)

    # Define initial_state as |1>
    initial_state = [0, 1]

    # Define state |q_0>
    initial_state = [1 / sqrt(2), 1j / sqrt(2)]

    # Apply initialisation operation to the 0th qubit
    qc.initialize(initial_state, 0)

    draw_quantum_circuit(qc, draw_unitary=False)


def initialize_with_own_utils():
    qc = QuantumCircuit(1)
    bit_to(qc, 0, State.y1)
    # qc.measure_all()  # This will collapse our state to either |0> o |1>
    draw_quantum_circuit(qc, draw_unitary=False, draw_histogram=True)


def exercise_1_3_1():
    qc = QuantumCircuit(1)
    initial_state = normalize([1, sqrt(2)])
    initial_state = normalize([1, 1j * sqrt(2)])
    print(initial_state)
    qc.initialize(initial_state, 0)
    draw_quantum_circuit(qc, draw_unitary=False, draw_histogram=True)


def draw_bloch_spheres():
    qc = QuantumCircuit(1)

    states = [
        [1, 0],
        [0, 1],
        [1, 1],
        [1, -1j],
        [1j, 1],
    ]
    for state in states:
        qc.initialize(normalize(state))
        draw_quantum_circuit(qc, draw_unitary=False)


if __name__ == '__main__':
    # initialize()
    # initialize_with_own_utils()
    # exercise_1_3_1()
    draw_bloch_spheres()
