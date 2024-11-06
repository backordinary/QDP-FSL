# https://github.com/erenaykrcn/qiskit-projects/blob/dcf0be7a104762bc53e73e5443fe491ef2a0c2ce/basic/teleportation.py
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.visualization import plot_bloch_multivector
from multipartite import generate_zero_zero_bell_state
from single_qubit import get_state_vector
from math import sqrt


def teleport(qc, q, i, j, crx, crz):
	# Takes a three qubit QR, whose first qubit is the 
	# qubit to be teleported and the other two are in 
	# source-destionation points

	generate_zero_zero_bell_state(qc, i, j)
	qc.barrier()

	# Bell Measurement:
	qc.cx(q, i)
	qc.h(q)
	qc.barrier()
	qc.measure([q, i], [crz[0], crx[0]])
	qc.barrier()
	# SV of the q1 and q2 are collapsed, therefore
	# delivers either [1,0] or [0,1]. We projected 
	# the wave function of the input q1-q2 state onto
	# the orthonormal basis of bell states, then measured
	# on the computational basis. Result corresponds to 
	# psi^(ij) bell state.

	# Getting the psi from collapsed state:
	qc.x(j).c_if(crx, 1) # execute if classical bit crx auf 1 gesetzt.
	qc.z(j).c_if(crz, 1)

	return qc


qr = QuantumRegister(3)
crz, crx = ClassicalRegister(1, name="crz"), ClassicalRegister(1, name="crx")
qc = QuantumCircuit(qr, crx, crz)
qc.initialize([1/sqrt(2), 1j/sqrt(2)], 0)

sv = get_state_vector(qc)
fig = plot_bloch_multivector(sv)
fig.savefig('unteleported_sv.png')

sv = get_state_vector(teleport(qc, 0, 1, 2, crx, crz))
fig = plot_bloch_multivector(sv)
fig.savefig('teleported_sv.png')

