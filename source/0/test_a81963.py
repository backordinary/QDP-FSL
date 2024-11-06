# https://github.com/1chooo/Quantum-Oracle/blob/2a2f8395c17129341c485d5345aa63c2be474f7c/test.py
"""
Test Script
"""

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

qrx = QuantumRegister(1, 'q0')
qry = QuantumRegister(1, 'q1')
cr = ClassicalRegister(1, 'c')

qc = QuantumCircuit(qrx, qry, cr)
qc.h(qrx)
qc.x(qry)
qc.h(qry)
qc.barrier()
qc.x(qry)
qc.barrier()
qc.h(qrx)
qc.h(qry)
qc.measure(qrx, cr)
qc.draw("mpl")

from qiskit import execute
from qiskit.providers.aer import AerSimulator
from qiskit.visualization import plot_histogram

sim = AerSimulator()
job = execute(qc, backend = sim, shots = 1000)
result = job.result()
counts = result.get_counts(qc)
print("Counts: ", counts)
plot_histogram(counts)