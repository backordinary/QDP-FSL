# https://github.com/erenaykrcn/qiskit-projects/blob/dcf0be7a104762bc53e73e5443fe491ef2a0c2ce/dj-algorithm/oracle.py
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.aqua.operators import X, Y, Z, I, CX, T, H, S, PrimitiveOp

import sys

sys.path.append("../basic")
from single_qubit import get_state_vector
from multipartite import figure_bloch_sphere


def implement_constant_oracle(qc, x, y, n):
	# Takes n qubit input Qubit Register x

	f = 0

	if f == 1:
		qc.x(y[0])
	elif f == 0:
		return qc
	else:
		raise ValueError("f must be either 0 or 1")
	return qc

def implement_balanced_oracle(qc, x, y, n):
	# bstring chracterises the off-set of the oracle
	# For exp: bstring = '101': Even number of 1s, odd
	# number of 0s: Two 1s and One 0 or Zero 1 and Three 0s
	# Off-set: {000, 110, 101, 011}
	#
	# Note: A possible bstring is equal to the other bstrings
	# in its off set cluster.

	bstring = ''
	for i in range(n):
		bstring += '0'

	# Apply X gates before the CNOTs
	#for i in range(len(bstring)):
	#	if bstring[i] == '1':
	#		qc.x(i)

	# Apply CNOTs
	for xi in range(n):
		qc.cx(x[xi], y[0])

	# Apply X Gates after the CNOTs 
	for i in range(n):
		if bstring[i] == '1':
			qc.x(i)
	return qc.to_gate() # gates can be appended to the QCs.
