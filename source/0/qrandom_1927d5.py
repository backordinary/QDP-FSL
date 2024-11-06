# https://github.com/nrhawkins/qk-learn/blob/8a3415d1233368c0b2cb2c93e49e8cfb984a0f66/python/qrandom.py
from qiskit import QuantumCircuit, transpile, assemble, Aer, IBMQ, execute
from qiskit.visualization import plot_bloch_multivector, plot_histogram

qc = QuantumCircuit(1,1)
qc.h(0)
qc.measure(0,0)

print("coin flip:")

qc.draw('mpl')

backend = Aer.get_backend('qasm_simulator')
counts = execute(qc, backend, shots = 1000).result().get_counts()

print("counts:", counts)

plot_histogram(counts)


