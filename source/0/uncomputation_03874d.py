# https://github.com/mcrl/quantum-benchmark/blob/37eb9e4256b08baabc97ef86992de9166211ba66/abstract/uncomputation.py
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import HGate, CXGate

def uncomputation(circ, c, m):
	u = c.inverse()
	u.name = 'uncomputation'
	circ.append(u, m)

def computation(circ, gates, qubits):
	inverse_map = []
	qubit_map = {}
	i = 0
	for qlist in qubits:
		for q in qlist:
			if q.index not in qubit_map:
				qubit_map[q.index] = i
				inverse_map.append(q)
				i += 1

	c = QuantumCircuit(len(qubit_map), name='computation')
	for g, qlist in zip(gates, qubits):
		print([qubit_map[q.index] for q in qlist])
		c.append(g, [qubit_map[q.index] for q in qlist])


	circ.append(c, inverse_map)
	return c, inverse_map
	

if __name__ == "__main__":
	num_qubits = 4
	qr = QuantumRegister(num_qubits, 'q')
	circ = QuantumCircuit(qr)

	gates = [HGate(), CXGate()]
	qubits = [[qr[0]], [qr[0], qr[2]]]

	c, m = computation(circ, gates, qubits)
	uncomputation(circ, c, m)
	print(circ)
