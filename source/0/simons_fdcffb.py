# https://github.com/mdschatz/qiskit/blob/929dd2dc5a68b69a06ea474998e9f973614bfec2/simons.py
import click
from matrix_util import null_space
import numpy as np
from qiskit import(
  QuantumCircuit,
  execute,
  Aer)


# From Fig 15 Intro to Quantum Computing, Without the Physics
def simon_circuit(Uf, n_qbits):
	circuit = QuantumCircuit(2*n_qbits, n_qbits)
	q_registers = list(range(2*n_qbits))
	m_registers = list(range(n_qbits))

	# Add a H gate on top qubits
	for i in range(n_qbits):
		circuit.h(i)
	circuit.unitary(Uf, q_registers)
	for i in range(n_qbits):
		circuit.h(i)

	# Map the quantum measurement to the classical bits
	circuit.measure(m_registers, m_registers)

	circuit.draw(filename='images/simon_circuit.png', output='mpl')
	return circuit


# From Section 4.3 Intro to Quantum Computing, Without the Physics
def simons_alg(Uf, n_qbits):
	circuit = simon_circuit(Uf, n_qbits)
	simulator = Aer.get_backend('qasm_simulator')

	E = np.random.rand(0,n_qbits)
	nIter = 0
	a_vec = None
	while a_vec is None and nIter < (10 * n_qbits):
		# Execute Simon's circuit
		result = execute(circuit, simulator, shots=8192).result()
		distribution = result.get_counts(circuit)
		k = max(distribution, key=distribution.get)
		count_frac = 1.0 * distribution[k] / sum(distribution.values())

		# Select k candidate
		k = np.fromstring(str(np.char.join(' ', k)), dtype=int, sep=' ')
		E = np.vstack([E, k])
		E = np.unique(E, axis=0)

		# Determine whether unique solution exists for 'a'
		# ns = scipy.linalg.null_space(E)
		ns = null_space(E)
		ns = ns % 2

		# Debugging
		# print(f"\n\niter {nIter}")
		# print(distribution)
		# print(f"max K found: {k} fraction: {count_frac}")
		# print("\nE")
		# print(E)
		# print("\nNull-space")
		# print(ns)

		if ns.shape[1] == 1:
			a_vec = ns[:,0]
		elif len(distribution.keys()) == 1 and all(x == 0 for x in k):
			a_vec = k
		nIter += 1
	return a_vec


# Created from 6.2 Intro to Quantum Computing, Without the Physics
def circuit_flip_but_bit(n_qbits, t_bit):
	circuit = QuantumCircuit(2*n_qbits)

	for i in range(n_qbits):
		if i != t_bit:
			# if i % 2 == 0:
			# 	circuit.x(i)
			circuit.cx(i, n_qbits + i)
			# if i % 2 == 0:
			# 	circuit.x(i)
	circuit.draw(filename='images/flip_but_bit_circuit.png', output='mpl')
	return circuit


# Note: Must be updated to reflect circuit_flip_but_bit logic
def verify_flip_but_bit(a_vec, t_bit):
	n_qbits = a_vec.shape[0]
	x = np.ones(n_qbits, dtype=int)

	z = (x - a_vec) % 2

	# Check that all entries are flipped except for t_bit
	for i in range(n_qbits):
		if i != t_bit and x[n_qbits-1-i] != z[n_qbits-1-i]:
			print("Test failed")
			return
		elif i == t_bit and x[n_qbits-1-i] == z[n_qbits-1-i]:
			print("Test failed")
			return
	return print("Test success")


# From https://quantumcomputinguk.org/tutorials/how-to-obtain-the-unitary-matrix-of-a-circuit-in-qiskit-with-code
def get_Uf(n_qbits, t_bit):
	circuit = circuit_flip_but_bit(n_qbits, t_bit)
	print(circuit)


	unitary_simulator = Aer.get_backend('unitary_simulator')
	job = execute(circuit, unitary_simulator, shots=8192)
	result = job.result()
	return result.get_unitary(circuit, 6)
	

@click.command()
@click.option("--n", type=int, help="Number of q-bits.", default=3)
@click.option("--t", type=int, help="Omitted bit during flip.", default=0)
def main(n, t):
	assert n > t

	Uf = get_Uf(n, t)
	a_vec = simons_alg(Uf, n)

	# Debugging
	# print("A")
	# print(a_vec)

	verify_flip_but_bit(a_vec, t)


if __name__ == "__main__":
	main()