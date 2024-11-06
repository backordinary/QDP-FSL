# https://github.com/mcrl/quantum-benchmark/blob/d04eb859e5ea0034c55075e27d9053f8456981f8/qiskit/iqft.py
import numpy as np
from qiskit import QuantumCircuit

def inverse_qft(num_qubits):
	circ = QuantumCircuit(num_qubits)

	for q in range(num_qubits):
		circ.h(q)
		for p in range(q+1, num_qubits):
			circ.cp(np.pi/2**(num_qubits-p), q, p) 

	for q in range(num_qubits//2):
	 	 circ.swap(q, num_qubits-q-1)
	
	return circ.inverse();






if __name__ == "__main__":
	num_qubits = 4
	print(inverse_qft(num_qubits))
