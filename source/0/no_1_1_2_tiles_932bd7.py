# https://github.com/jamesgopsill/ICED21-Quantum-Design/blob/bbb3d60639f0dbb81aa18165647eb4a0769d1a26/qasm/circuits/no_1_1_2_tiles.py

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

def no_1_1_2_tiles() -> QuantumCircuit:
	
	# Initialise the quantum register
	qreg = QuantumRegister(6, "q")
	creg = ClassicalRegister(4, "c")

	# Initialise the circuit
	circuit = QuantumCircuit(qreg, creg)

	circuit.h(qreg[0])
	circuit.h(qreg[1])
	circuit.h(qreg[2])
	circuit.h(qreg[3])

	circuit.ccx(qreg[0], qreg[1], qreg[4])
	circuit.cx(qreg[4], qreg[0])

	circuit.barrier()

	circuit.ccx(qreg[2], qreg[3], qreg[5])
	circuit.cx(qreg[5], qreg[2])

	circuit.barrier()

	circuit.measure(0, 0)
	circuit.measure(1, 1)
	circuit.measure(2, 2)
	circuit.measure(3, 3)

	return circuit
