# https://github.com/slowy07/quantum_computing/blob/0fefa6e53066b04013ed114c894d672c6d68b5a5/testing.py
from qiskit import QuantumCircuit

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()
qc.draw()
