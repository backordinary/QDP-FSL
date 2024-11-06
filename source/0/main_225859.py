# https://github.com/brodkemd/UC_Quantum_Lab/blob/acdba883487f47e31dbb4d4fbc4bb7e102123cd7/templates/main.py
from UC_Quantum_Lab.commands import state, display, counts
from UC_Quantum_Lab.layout import invert, horizontal_invert, vertical_invert, custom
from qiskit import QuantumCircuit

qc = QuantumCircuit(1, 1)
qc.h(0)
state(qc)
qc.measure_all()
display(qc)