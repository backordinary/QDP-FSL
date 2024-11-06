# https://github.com/erenaykrcn/qiskit-projects/blob/dcf0be7a104762bc53e73e5443fe491ef2a0c2ce/grover/oracle.py
from qiskit import QuantumCircuit
import numpy as np
from qiskit.quantum_info import Operator


def phase_oracle(n, indices):
	"""
		If we want to implement it with known gates,
		to mark for exp 101 state, apply cz to 0,2.
	"""

	qc = QuantumCircuit(n, name="Uf - Phase Oracle")
	matrix = np.identity(2**n)
	for index in indices:
		matrix[index, index] = -1
	qc.unitary(Operator(matrix), range(n))
	return qc


def diffuser(nqubits):
    qc = QuantumCircuit(nqubits)
    # Apply transformation |s> -> |00..0> (H-gates)
    for qubit in range(nqubits):
        qc.h(qubit)
    # Apply transformation |00..0> -> |11..1> (X-gates)
    for qubit in range(nqubits):
        qc.x(qubit)
    # Do multi-controlled-Z gate
    qc.h(nqubits-1)
    qc.mct(list(range(nqubits-1)), nqubits-1)  # multi-controlled-toffoli
    qc.h(nqubits-1)
    # Apply transformation |11..1> -> |00..0>
    for qubit in range(nqubits):
        qc.x(qubit)
    # Apply transformation |00..0> -> |s>
    for qubit in range(nqubits):
        qc.h(qubit)
    # We will return the diffuser as a gate
    U_s = qc.to_gate()
    U_s.name = "diffuser"
    return U_s



def diffuser_(n):
	qc = QuantumCircuit(n, name="Diffuser")
	qc.h(range(n))

	matrix = np.identity(2**n)
	for index in range(2**n):
		if index != 0:
			matrix[index, index] = -1
	qc.unitary(Operator(matrix), range(n))

	qc.h(range(n))
	return qc
