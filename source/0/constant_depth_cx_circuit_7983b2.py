# https://github.com/QuantMarkFramework/circuit-analyser/blob/b24ddefb6b794388b21ac4a46273cc6a851123b8/analyser/extra/constant_depth_CX_circuit.py
from random import random, shuffle
from pytket import Circuit, OpType
from qiskit import QuantumCircuit
from numpy import pi


def constant_depth_CX_circuit(
	qubits: int,
	layers: int,
	library: str,
	single_qubit_gates: bool = True,
):
	if not isinstance(library, str):
		raise ValueError("Library parameter has to be a string")
	elif library.lower() == "pytket":
		return _tket_version(qubits, layers, single_qubit_gates)
	elif library.lower() == "qiskit":
		return _qiskit_version(qubits, layers, single_qubit_gates)
	else:
		raise ValueError("Library has to be given as 'pytket' or 'qiskit'.")


def _r(val: int):
	return val * random()


def _tket_version(qubits: int, layers: int, single_qubit_gates: bool):
	q = list(range(qubits))
	circuit: Circuit = Circuit(qubits)
	for _ in range(layers):
		shuffle(q)
		for qubit in q:
			circuit.add_gate(OpType.U3, [_r(4), _r(4), _r(4)], [qubit])
		for i in range(qubits // 2):
			circuit.CX(control_qubit=q[i * 2], target_qubit=q[i * 2 + 1])
	return circuit


def _qiskit_version(qubits: int, layers: int, single_qubit_gates: bool):
	q = list(range(qubits))
	circuit: QuantumCircuit = QuantumCircuit(qubits)
	for _ in range(layers):
		shuffle(q)
		for qubit in q:
			circuit.u(_r(4 * pi), _r(4 * pi), _r(4 * pi), qubit)
		for i in range(qubits // 2):
			circuit.cx(control_qubit=q[i * 2], target_qubit=q[i * 2 + 1])
	return circuit
