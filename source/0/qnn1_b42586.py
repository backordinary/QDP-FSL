# https://github.com/PhilWun/MAProjekt/blob/3b115e2a4c1c9f0419bbc515f2318bd50794d268/src/qk/QNN1.py
from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister
from qiskit.circuit import Parameter, Instruction


def create_qiskit_circuit(param_prefix: str, num_qubits: int) -> QuantumCircuit:
	"""
	Implements circuit A from J. Romero, J. P. Olson, and A. Aspuru-Guzik, “Quantum autoencoders for efficient compression
	of quantum data,” Quantum Sci. Technol., vol. 2, no. 4, p. 045001, Dec. 2017, doi: 10.1088/2058-9565/aa8072.

	:param param_prefix:
	:param num_qubits:
	:return:
	"""
	qr = QuantumRegister(num_qubits)
	cr = ClassicalRegister(num_qubits)
	unit_cell = QuantumCircuit(qr, cr)
	idx = 1

	for i in range(num_qubits - 1):
		for j in range(num_qubits - 1 - i):
			_add_two_qubit_gate(param_prefix + "U" + str(idx) + "_", unit_cell, qr[j], qr[j + i + 1])
			idx += 1

	unit_cell.measure(qr, cr)

	return unit_cell


def _add_two_qubit_gate(param_prefix: str, qc: QuantumCircuit, q1, q2):
	"""
	Adds a general two-qubit gate to the specified quantum circuit.
	Implements a general two-qubit gate as seen in F. Vatan and C. Williams, “Optimal Quantum Circuits for General
	Two-Qubit Gates,” Phys. Rev. A, vol. 69, no. 3, p. 032315, Mar. 2004, doi: 10.1103/PhysRevA.69.032315.

	:param param_prefix: prefix for the parameters
	:param qc: the quantum circuit where the gate should be added
	:param q1: first input qubit for the gate
	:param q2: second input qubit for the gate
	"""

	qc.u(
		Parameter(param_prefix + "a1_a"),
		Parameter(param_prefix + "a1_b"),
		Parameter(param_prefix + "a1_c"),
		q1)

	qc.u(
		Parameter(param_prefix + "a2_a"),
		Parameter(param_prefix + "a2_b"),
		Parameter(param_prefix + "a2_c"),
		q2
	)

	qc.cnot(q2, q1)

	qc.rz(Parameter(param_prefix + "t1"), q1)
	qc.ry(Parameter(param_prefix + "t2"), q2)

	qc.cnot(q1, q2)

	qc.ry(Parameter(param_prefix + "t3"), q2)

	qc.cnot(q2, q1)

	qc.u(
		Parameter(param_prefix + "a3_a"),
		Parameter(param_prefix + "a3_b"),
		Parameter(param_prefix + "a3_c"),
		q1
	)

	qc.u(
		Parameter(param_prefix + "a4_a"),
		Parameter(param_prefix + "a4_b"),
		Parameter(param_prefix + "a4_c"),
		q2
	)


def main():
	create_qiskit_circuit("", 3)


if __name__ == "__main__":
	main()
