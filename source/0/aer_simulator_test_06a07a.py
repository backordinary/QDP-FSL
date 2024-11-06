# https://github.com/pontaGC/IBMQ/blob/42adfbe7d86c7bb454abfd4e0967eef0da759194/Aer_Simulator_test.py
# -*- coding: utf-8 -*-

from qiskit import QuantumCircuit
from qiskit import execute
import numpy as np
import matplotlib.pyplot as plt

import Aer_Simulator_Type as AerSimType
import Aer_Simulator_Provider


#
# state vector simulator
#

def test_state_vector_simulator():
    qc = QuantumCircuit(2)

    qc.h(0)
    qc.cx(0, 1)

    backend = Aer_Simulator_Provider.get_backend(AerSimType.Type.StateVector)
    results = execute(qc, backend=backend).result()
    state_vec = results.get_statevector(qc)

    print(state_vec)
    # Output:
    # [0.70710678 + 0.j, 0.+ 0.j, 0. + 0.j,                  0.70710678 + 0.j]
    # => 1/√2 (|00> + |11>)

#
# unitary simulator
#
def test_unitary_simulator():
    qc = QuantumCircuit(2)

    qc.h(0)
    qc.cx(0, 1)

    backend = Aer_Simulator_Provider.get_backend(AerSimType.Type.Unitary)
    results = execute(qc, backend=backend).result()
    unitary_matrix = results.get_unitary(qc)

    #  value round for easy to see
    print(np.round(unitary_matrix, 4))

    # Output
    # [0.7071 + 0.j  0.7071 - 0.j  0.     + 0.j  0.     + 0.j]
    # [0.     + 0.j  0.     + 0.j  0.7071 + 0.j -0.7071 + 0.j]
    # [0.     + 0.j  0.     + 0.j  0.7071 + 0.j  0.7071 - 0.j]
    # [0.7071 + 0.j -0.7071 + 0.j  0.     + 0.j  0.     + 0.j]

#
# qasm simulator
#
def test_qasm_simulator():
    # 2-argument is classic register size
    qc = QuantumCircuit(2, 2)

    qc.h(0)
    qc.cx(0, 1)

    # measure({observable Q-bit}, {classic register index to store observed Q-bit})
    qc.measure(0, 0)
    qc.measure(1, 1)

    qc.draw(output='mpl')

    backend = Aer_Simulator_Provider.get_backend(AerSimType.Type.Qasm)
    # number of samplings
    observable_shots = 1024
    results = execute(qc, backend=backend, shots=observable_shots).result()
    answer = results.get_counts()

    print(answer)
    # Output
    # {'00': 486, '11': 538}
    # Left: State
    # Right: Observed count

    plt.show()

if __name__ == '__main__':
    test_state_vector_simulator()
    test_unitary_simulator()
    test_qasm_simulator()
