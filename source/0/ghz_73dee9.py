# https://github.com/Gcuencam/full-tomography/blob/331217f6a97038f0e2e74cd63a230c6aec1ea583/src/states/ghz.py
# -*- coding: utf-8 -*-

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


def build(qc, ref_pos, n):
    qc = qc.copy()
    qc.h(ref_pos)
    # Add a CNOT between each qbit and the next:
    for i in range(ref_pos, n - 1):
        qc.cx(i, i + 1)
    return qc


def get_ghz_state_vector(qc_size):
    circuit = QuantumCircuit(qc_size, qc_size)
    ghz_qc = build(circuit, 0, qc_size)
    psi = Statevector.from_instruction(ghz_qc)
    return np.array(psi)


if __name__ == '__main__':
    qc_size = 2
    print(get_ghz_state_vector(qc_size))
