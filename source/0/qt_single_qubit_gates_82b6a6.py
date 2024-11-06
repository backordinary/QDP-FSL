# https://github.com/nrhawkins/qalgs/blob/a0aa4dd3db5da77528d723ff15d2d402ea8022c6/circuits/qt_single_qubit_gates.py

from qiskit import QuantumCircuit, execute, Aer
from qiskit.visualization import plot_histogram

n = 8

n_q = n
n_b = n

qc_output = QuantumCircuit(n_q, n_b)

for j in range(n):
    qc_output.measure(j,j)

qc_output.draw()


