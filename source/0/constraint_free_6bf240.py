# https://github.com/jamesgopsill/ICED21-Quantum-Design/blob/bbb3d60639f0dbb81aa18165647eb4a0769d1a26/statevectors/circuits/constraint_free.py
from qiskit import QuantumCircuit, QuantumRegister

def constraint_free() -> QuantumCircuit:
	qreg = QuantumRegister(4, "q")

	circuit = QuantumCircuit(qreg)

	circuit.h(qreg[0])
	circuit.h(qreg[1])
	circuit.h(qreg[2])
	circuit.h(qreg[3])

	return circuit

