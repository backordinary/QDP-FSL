# https://github.com/jamesgopsill/ICED21-Quantum-Design/blob/bbb3d60639f0dbb81aa18165647eb4a0769d1a26/statevectors/circuits/no_overlap.py
from qiskit import QuantumCircuit, QuantumRegister



def no_overlap() -> QuantumCircuit:

	qreg = QuantumRegister(7, "q")

	circuit = QuantumCircuit(qreg)

	circuit.h(qreg[0])
	circuit.h(qreg[1])
	circuit.h(qreg[2])
	circuit.h(qreg[3])

	circuit.ccx(qreg[0], qreg[2], qreg[4])

	circuit.x(qreg[0])
	circuit.x(qreg[2])

	circuit.barrier()

	circuit.ccx(qreg[0], qreg[2], qreg[4])

	circuit.x(qreg[0])
	circuit.x(qreg[2])

	circuit.barrier()

	circuit.ccx(qreg[1], qreg[3], qreg[5])

	circuit.x(qreg[1])
	circuit.x(qreg[3])

	circuit.barrier()

	circuit.ccx(qreg[1], qreg[3], qreg[5])

	circuit.x(qreg[1])
	circuit.x(qreg[3])

	circuit.barrier()

	circuit.ccx(qreg[4], qreg[5], qreg[6])
	circuit.cx(qreg[6], qreg[3])

	return circuit


