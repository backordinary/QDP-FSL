# https://github.com/mistershashanksingh/Basic-Practice-Project/blob/80c0924aa62bde9d010a6436379e2dbc3a6924e8/PYTHON/QUANTUM/quntum/qiskit1.py
from qiskit import QuantumCircuit
# Try to run a sample
# This is quantum computing code__________
qc = QuantumCircuit(2)  # Here i creates a circuit with 2 qu bits
qc.h(0)  # A I do a H-gate on q0
qc.cx(0, 1)  # A I do a C-NOT on q1 which is gonna control by q0
qc.measure_all()
qc.draw()
