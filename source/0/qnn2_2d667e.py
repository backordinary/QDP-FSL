# https://github.com/PhilWun/MAProjekt/blob/3b115e2a4c1c9f0419bbc515f2318bd50794d268/src/qk/QNN2.py
from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister
from qiskit.circuit import Parameter
from qiskit.circuit.library import UGate, RZGate, RYGate


def create_qiskit_circuit(param_prefix: str, num_qubits: int) -> QuantumCircuit:
	"""
	Implements circuit B from J. Romero, J. P. Olson, and A. Aspuru-Guzik, “Quantum autoencoders for efficient compression
	of quantum data,” Quantum Sci. Technol., vol. 2, no. 4, p. 045001, Dec. 2017, doi: 10.1088/2058-9565/aa8072.

	:param param_prefix:
	:param num_qubits:
	:return:
	"""
	qr = QuantumRegister(num_qubits)
	cr = ClassicalRegister(num_qubits)
	unit_cell = QuantumCircuit(qr, cr)
	layer_idx = 0

	# first layer of single qubit rotations
	for i in range(num_qubits):
		unit_cell.u(
			Parameter(param_prefix + str(layer_idx) + "_" + str(i) + "a"),
			Parameter(param_prefix + str(layer_idx) + "_" + str(i) + "b"),
			Parameter(param_prefix + str(layer_idx) + "_" + str(i) + "c"),
			qr[i])

	unit_cell.barrier()

	layer_idx += 1

	# one layer of controlled ratations per qubit
	for i in range(num_qubits):
		for j in range(num_qubits):
			if i != j:
				# TODO: replace with CU3 gate
				unit_cell.append(
					RZGate(Parameter(param_prefix + str(layer_idx + i) + "_" + str(i) + "_" + str(j) + "a")).control(),
					[qr[i], qr[j]])
				unit_cell.append(
					RYGate(Parameter(param_prefix + str(layer_idx + i) + "_" + str(i) + "_" + str(j) + "b")).control(),
					[qr[i], qr[j]])
				unit_cell.append(
					RZGate(Parameter(param_prefix + str(layer_idx + i) + "_" + str(i) + "_" + str(j) + "c")).control(),
					[qr[i], qr[j]])

	unit_cell.barrier()
	layer_idx += num_qubits

	# last layer of single qubit rotations
	for i in range(num_qubits):
		unit_cell.u(
			Parameter(param_prefix + str(layer_idx) + "_" + str(i) + "a"),
			Parameter(param_prefix + str(layer_idx) + "_" + str(i) + "b"),
			Parameter(param_prefix + str(layer_idx) + "_" + str(i) + "c"),
			qr[i])

	unit_cell.barrier()
	unit_cell.measure(qr, cr)

	return unit_cell


def main():
	create_qiskit_circuit("", 3)


if __name__ == "__main__":
	main()
