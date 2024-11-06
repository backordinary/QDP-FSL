# https://github.com/QBugs/qsmells-study-data/blob/501f722cb67ee135fb6ea1748472d940b47eadec/samples/lpq/lpq-smelly.py
from qiskit import QuantumCircuit, transpile
from qiskit.providers.fake_provider import FakeVigo
backend = FakeVigo()

qc = QuantumCircuit(3, 3)
qc.h(0)
qc.cx(0,range(1,3))
qc.barrier()
qc.measure(range(3), range(3))
qc = transpile(qc, backend)

# ------------------------------------------------------------------------------

# Draw
qc.draw(output='text', filename='lpq-smelly.txt', justify='left')
qc.draw(output='latex_source', filename='lpq-smelly.tex', justify='left')
qc.draw(output='mpl', filename='lpq-smelly.pdf', justify='left', fold=-1)
qc.draw(output='mpl', filename='lpq-smelly-folded.pdf', justify='left')

from qiskit.visualization import plot_circuit_layout
fig = plot_circuit_layout(qc, backend, view='virtual')
fig.savefig('lpq-smelly-virtual-circuit.pdf')
fig = plot_circuit_layout(qc, backend, view='physical')
fig.savefig('lpq-smelly-physical-circuit.pdf')

from quantum_circuit_to_matrix import Justify, qc2matrix
qc2matrix(qc, Justify.left, 'lpq-smelly.csv')
