# https://github.com/mcrl/quantum-benchmark/blob/37eb9e4256b08baabc97ef86992de9166211ba66/abstract/modular.py
import math
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import HGate, CXGate

def modular(circ, a, power, num_qubits, num_ansillas):
	N = 2 ** num_ansillas - 1
	if math.gcd(a, N) != 1:
		raise "Illegal a" 
	print(N)
	U = QuantumCircuit(num_ansillas)

	M = N+1
	mat = np.identity(M)
	mat[a,a] = 0
	mat[a,N] = 1
	mat[N,a] = 1
	mat[N,N] = 0
	print(mat)
	for iteration in range(power):
		if a in [2,13]:
			U.swap(0,1)
			U.swap(1,2)
			U.swap(2,3)
		if a in [7,8]:
			U.swap(2,3)
			U.swap(1,2)
			U.swap(0,1)
		if a in [4, 11]:
			U.swap(1,3)
			U.swap(0,2)
		if a in [7,11,13]:
			for q in range(power):
				U.x(q)
	#U = U.to_gate()
	U.name = "%i^%i mod 15" % (a, power)
	print(U)
	return circ
	

if __name__ == "__main__":
	num_qubits = 8
	num_ansillas = 4
	a = 4

	circ = QuantumCircuit(num_qubits+num_ansillas, num_ansillas)
	for i in range(num_qubits):
		circ.h(i)
	circ.x(num_qubits+num_ansillas-1)
	print(modular(circ, a, 1, num_qubits, num_ansillas))

