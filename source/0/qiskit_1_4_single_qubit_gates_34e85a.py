# https://github.com/physicalattraction/quantum-computing-tutorial/blob/f88fe925c5988b61339cda25908eb587f1c9acd3/src/qiskit_1_4_single_qubit_gates.py
"""
https://qiskit.org/textbook/ch-states/single-qubit-gates.html
"""

from typing import List

from math import pi, sqrt
from qiskit import *
from qiskit.visualization import plot_bloch_multivector, plot_histogram

# These are the states along the Z, X and Y axis
# The direction is a human-readable indication where on the Block sphere they lie
from utils import STATES, draw_quantum_circuit, states_to_vector

a = 1 / sqrt(2)  # Normalization factor
state_z0 = [1, 0]  # |0> = [1, 0] = up
state_z1 = [0, 1]  # |1> = [0, 1] = down
state_x0 = [a, a]  # |+> = 1/sqrt(2) * [1, 1] = front
state_x1 = [a, -a]  # |-> = 1/sqrt(2) * [1, -1] = back
state_y0 = [a * 1j, a]  # |↺> = 1/sqrt(2) * [i, 1] = left
state_y1 = [-a * 1j, a]  # |↻> = 1/sqrt(2) * [-i, 1] = right


def show_qc(qc: QuantumCircuit):
    """
    Display the given state vector of the quantum circuit
    """

    # print(qc.draw())
    backend = Aer.get_backend('statevector_simulator')
    out = execute(qc, backend).result().get_statevector()
    plot_bloch_multivector(out).show()
    return out


def get_counts_after_measurement(qc: QuantumCircuit) -> List[int]:
    return execute(qc, Aer.get_backend('qasm_simulator')).result().get_counts()


def show_all_initial_states():
    """
    Draw the Bloch spheres for all 6 initial states
    """

    for initial_state in STATES:
        qc = QuantumCircuit(1, 1)
        qc.initialize(states_to_vector(initial_state), 0)
        draw_quantum_circuit(qc, draw_bloch_sphere=True)
        print('\n**********\n')


def x_gate():
    """
    Execute an X-gate operator on all 6 initial states and draw the Block spheres
    """

    for initial_state in STATES:
        qc = QuantumCircuit(1, 1)
        qc.initialize(states_to_vector(initial_state), 0)
        qc.x(0)
        draw_quantum_circuit(qc, draw_bloch_sphere=True)
        print('\n**********\n')


def y_gate():
    """
    Execute a Y-gate operator on all 6 initial states and draw the Block spheres
    """

    for initial_state in STATES:
        qc = QuantumCircuit(1, 1)
        qc.initialize(states_to_vector(initial_state), 0)
        qc.y(0)
        show_qc(qc)


def z_gate():
    """
    Execute a Z-gate operator on all 6 initial states and draw the Block spheres
    """

    for initial_state in [state_z0, state_z1, state_x0, state_x1, state_y0,
                          state_y1]:
        qc = QuantumCircuit(1, 1)
        qc.initialize(initial_state, 0)
        qc.z(0)
        show_qc(qc)


def h_gate():
    """
    Execute a Hadamard gate operator on all 6 initial states and draw the Block spheres

    The Hadamard gate can also be expressed as a 90º rotation around the Y-axis,
    followed by a 180º rotation around the X-axis. So, H = X Y^{1/2}

    Useful XY-decompositions are given by:
    H = X Y^{1/2}
    H = Y^{-1/2} X

    Useful YZ-decompositions are:
    H = Z Y^{-1/2}
    H = Y^{1/2} Z

    |0> -> |+>
    |1> -> |->
    |+> -> |0>
    |-> -> |1>
    |↺> -> |↻>
    |↻> -> |↺>
    """

    for initial_state in [state_z0, state_z1, state_x0, state_x1, state_y0,
                          state_y1]:
        qc = QuantumCircuit(1, 1)
        qc.initialize(initial_state, 0)
        qc.h(0)
        show_qc(qc)


def hxh_gate():
    """
    I have calculated that HZH = I. This checks whether that is correct
    """

    for initial_state in [state_z0, state_z1, state_x0, state_x1, state_y0,
                          state_y1]:
        qc = QuantumCircuit(1, 1)
        qc.initialize(initial_state, 0)
        qc.h(0)
        qc.x(0)
        qc.h(0)
        draw_quantum_circuit(qc, draw_circuit=False, draw_bloch_sphere=False,
                             draw_unitary=False)


def z_measurement(initial_state=None):
    """
    Measure in Z-basis of a |0> qubit
    """

    # Initialize quantum circuit with 1 quantum bit and 1 classical bit
    qc = QuantumCircuit(1, 1)

    # Set initial state
    if initial_state is None:
        # Along Z-axis:
        #   |0> = [1, 0]
        #   |1> = [0, 1]
        # Along X-axis
        #   |+> = 1/sqrt(2) * [1, 1]
        #   |-> = 1/sqrt(2) * [1, -1]
        # Along Y-axis:
        #   |↺> = 1/sqrt(2) * [i, 1]
        #   |↻> = 1/sqrt(2) * [-i, 1]

        initial_state = [1, 0]
    qc.initialize(initial_state, 0)

    # Measure the 0th qubit and store the result in the 0th classical bit
    # qc.measure(0, 0)
    show_qc(qc)


def x_measurement():
    def perform_x_measurement(qc: QuantumCircuit, qubit: int,
                              cbit: int) -> QuantumCircuit:
        """
        Measure 'qubit' in the X-basis, and store the result in 'cbit'
        """

        qc.h(qubit)
        qc.measure(qubit, cbit)
        qc.h(qubit)
        return qc

    qc = QuantumCircuit(1, 1)
    initial_state = state_z0
    qc.initialize(initial_state, 0)

    perform_x_measurement(qc, 0, 0)  # measure qubit 0 to classical bit 0
    counts = get_counts_after_measurement(qc)
    plot_histogram(counts).show()


def rz_gate():
    """
    Execute a Rz-gate operator on all 6 initial states and draw the Block spheres
    """

    for initial_state in [state_z0, state_z1, state_x0, state_x1, state_y0,
                          state_y1]:
        for angle in [pi / 4, pi / 2]:
            qc = QuantumCircuit(1, 1)
            qc.initialize(initial_state, 0)
            qc.rz(angle, 0)
            show_qc(qc)


def s_gate():
    # An S-gate is the square root of a Z-gate, equivalent to Rz(pi/2)

    # Original: X
    qc = QuantumCircuit(1, 1)
    qc.initialize(state_x0, 0)
    show_qc(qc)

    # Rz(pi/2)
    qc.initialize(state_x0, 0)
    qc.rz(pi / 2, 0)
    show_qc(qc)

    # S
    qc.initialize(state_x0, 0)
    qc.s(0)
    show_qc(qc)

    # S dagger
    qc.initialize(state_x0, 0)
    qc.sdg(0)
    show_qc(qc)


def t_gate():
    # A T-gate is the fourth power root of a Z-gate, equivalent to Rz(pi/4)

    # Original: X
    qc = QuantumCircuit(1, 1)
    qc.initialize(state_x0, 0)
    show_qc(qc)

    # Rz(pi/4)
    qc.initialize(state_x0, 0)
    qc.rz(pi / 4, 0)
    show_qc(qc)

    # T
    qc.initialize(state_x0, 0)
    qc.t(0)
    show_qc(qc)

    # T dagger
    qc.initialize(state_x0, 0)
    qc.tdg(0)
    show_qc(qc)


if __name__ == '__main__':
    # show_all_initial_states()
    x_gate()
    # y_gate()
    # z_gate()
    # h_gate()
    # hxh_gate()
    # x_measurement()
    # rz_gate()
    # s_gate()
    # t_gate()
