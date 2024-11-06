# https://github.com/erenaykrcn/qiskit-projects/blob/dcf0be7a104762bc53e73e5443fe491ef2a0c2ce/sudoku/taffoli.py
from qiskit import QuantumCircuit, QuantumRegister

import sys
sys.path.append("../basic")
from single_qubit import get_state_vector

qc = QuantumCircuit(4)

qc.h(range(3))
qc.mct([0, 1, 2], 3)
print(get_state_vector(qc))
print(qc.draw())