# https://github.com/SEQUOIA-Demonstrators/zne_sequoia/blob/20709a1b07cba3b685e99539968b0e3714b8cc6a/algorithm.py
import numpy as np
from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister

def algo(theta=0):		
	"""
	This algorithm is more or less completly taken from the
	qiskit textbook: https://qiskit.org/textbook/ch-applications/hhl_tutorial.html
	"""
	t =np.pi *3/4
	nqubits = 4  # Total number of qubits
	nb = 1  # Number of qubits representing the solution
	nl = 2  # Number of qubits representing the eigenvalues
	cbit = 1

	a = 1  # Matrix diagonal
	b = -1/3  # Matrix off-diagonal (Nebendiagonaleinträge)
	# Initialise the quantum and classical registers
	qr = QuantumRegister(nqubits)
	cr = ClassicalRegister(cbit)
	# Create a Quantum Circuit
	qc = QuantumCircuit(qr,cr)

	qrb = qr[0:nb] # Qubits mit Lösung
	qrl = qr[nb:nb+nl] # Qubits für Eigenwerte
	qra = qr[nb+nl:nb+nl+1] # Hilfsqubit

	# State preparation.  
	qc.ry(2*theta, 0) #dreht in Bloch um theta um y-Achse


	# QPE with e^{iAt}
	for qu in qrl: #h auf alle Qubits in qrl (warum nicht qc.h(qrl))
		qc.h(qu) 

	qc.p(a*t, 1) #?
	qc.p(a*t*2, 2) #?
	qc.u(b*t, -np.pi/2, np.pi/2, 0) #?

	# Controlled e^{iAt} on \lambda_{1}:
	params=b*t

	qc.p(np.pi/2,0)
	qc.cx(1,0)

	qc.ry(params,0)
	qc.cx(1,0)

	qc.ry(-params,0)
	qc.p(3*np.pi/2,0)

	# Controlled e^{2iAt} on \lambda_{2}:
	params = b*t*2

	qc.p(np.pi/2,0)
	qc.cx(2,0)

	qc.ry(params,0)
	qc.cx(2,0)

	qc.ry(-params,0)
	qc.p(3*np.pi/2,0)

	# Inverse QFT
	qc.h(2)
	qc.rz(-np.pi/4,2)
	qc.cx(1,2)

	
	qc.rz(np.pi/4,2)
	qc.cx(1,2)

	qc.rz(-np.pi/4,1)
	qc.h(1)

	# Eigenvalue rotation
	t1=(-np.pi +np.pi/3 - 2*np.arcsin(1/3))/4
	t2=(-np.pi -np.pi/3 + 2*np.arcsin(1/3))/4
	t3=(np.pi -np.pi/3 - 2*np.arcsin(1/3))/4
	t4=(np.pi +np.pi/3 + 2*np.arcsin(1/3))/4

	qc.cx(2,3)

	qc.ry(t1,3)
	qc.cx(1,3)

	qc.ry(t2,3)
	qc.cx(2,3)

	qc.ry(t3,3)
	qc.cx(1,3)

	qc.ry(t4,3)

	qc.measure([qr[3]],[0])
	return qc
	
def eval_counts(counts):
	"""Gives back the norm of the vector for given counts (HHL-algorithm)"""
	t = np.pi *3/4
	C = 1/4 
	p1 = 0
	shots = 0
	for i in counts:
		shots += counts[i]
		if i[0] == "1":
			p1 += counts[i]
	p1 = p1/shots
	xnorm = np.sqrt(t**2 * p1 / (4*np.pi**2 * C**2))
	return xnorm
