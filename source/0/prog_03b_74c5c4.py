# https://github.com/1chooo/Quantum-Algorithm/blob/c61170ed839c4f1f92880b28cc4c69fc877ca5ae/CH05/prog_03b.py
from qiskit import execute
from qiskit.providers.aer import AerSimulator
from qiskit.visualization import plot_histogram
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit

qrx = QuantumRegister(3, 'x')
qry = QuantumRegister(1, 'y')
cr = ClassicalRegister(3, 'c')
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


sim = AerSimulator()
job = execute(qc, backend = sim, shots = 1000)
result = job.result()
counts = result.get_counts(qc)

print("Counts: ", counts)
plot_histogram(counts)