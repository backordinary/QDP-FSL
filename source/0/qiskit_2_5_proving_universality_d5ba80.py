# https://github.com/physicalattraction/quantum-computing-tutorial/blob/4586c5702de0326f766a35953c6b9edc19eabe30/src/qiskit_2_5_proving_universality.py
"""
https://qiskit.org/textbook/ch-gates/proving-universality.html
"""

import warnings
from math import pi

from qiskit import QuantumCircuit

from utils import draw_quantum_circuit


def cnot_conjugation():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        q = QuantumCircuit(2)
        q.cnot(1, 0)
        q.x(1)
        q.cnot(1, 0)
        draw_quantum_circuit(q, draw_circuit=False, draw_bloch_sphere=False, draw_final_state=False)


def build_unitary():
    """
    Build the unitary U=ei(aX⊗X⊗X+bZ⊗Z⊗Z)
    """

    theta = pi / 3  # Agnle determined by the gate we have at our disposal

    # Create the unitary eiθ2X⊗X⊗X from a single qubit Rx(θ) and two controlled-NOTs
    qc = QuantumCircuit(3)
    qc.cx(0, 2)
    qc.cx(0, 1)
    qc.rx(theta, 0)
    qc.cx(0, 1)
    qc.cx(0, 2)

    # Do the same for eiθ2Z⊗Z⊗Z with a few Hadmards
    qc.h(0)
    qc.h(1)
    qc.h(2)
    qc.cx(0, 2)
    qc.cx(0, 1)
    qc.rx(theta, 0)
    qc.cx(0, 1)
    qc.cx(0, 2)
    qc.h(2)
    qc.h(1)
    qc.h(0)

    draw_quantum_circuit(qc, draw_unitary=True)


if __name__ == '__main__':
    # cnot_conjugation()
    build_unitary()
