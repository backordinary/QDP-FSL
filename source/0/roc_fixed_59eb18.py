# https://github.com/QBugs/qsmells-study-data/blob/501f722cb67ee135fb6ea1748472d940b47eadec/samples/roc/roc-fixed.py
from qiskit import QuantumCircuit

qc = QuantumCircuit(3, 3) # 3 Quantum and 3 Classical registers

hadamard = QuantumCircuit(1, name='H')
hadamard.h(0)

measureQubit = QuantumCircuit(1, 1, name='M')
measureQubit.measure(0, 0)
qc.append(hadamard, [0])
qc.append(hadamard, [1])
qc.append(hadamard, [2])
qc.append(measureQubit, [0], [0])
qc.append(measureQubit, [1], [1])
qc.append(measureQubit, [2], [2])
qc.barrier()

qc.repeat(4)

# ------------------------------------------------------------------------------

from qiskit import transpile

# Transpile
qc = transpile(qc, basis_gates=['u1', 'u2', 'u3', 'rz', 'sx', 'x', 'cx', 'id'], optimization_level=0)

# Draw
qc.draw(output='text', filename='roc-fixed.txt', justify='left')
qc.draw(output='latex_source', filename='roc-fixed.tex', justify='left')
qc.draw(output='mpl', filename='roc-fixed.pdf', justify='left', fold=-1)
qc.draw(output='mpl', filename='roc-fixed-folded.pdf', justify='left')

from quantum_circuit_to_matrix import Justify, qc2matrix
qc2matrix(qc, Justify.left, 'roc-fixed.csv')
