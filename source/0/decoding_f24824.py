# https://github.com/eolocon/qkd_project/blob/ea2e69ccc69d165ebdd539e1a127347203a8cb7f/src/e91/decoding.py
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, transpile
from qiskit.circuit.library import IGate, HGate, RYGate
from math import pi
from bitstring import Bits

class Decoder:
	"""Decoder for E91 protocol.

	Creates the raw keys for the two partecipants (Alice and Bob) to the protocol.

	"""

	def decode(self, a_bases, b_bases, quantum_states):
		"""Creates the raw keys for Alice and Bob. 

		The i-th bit of the key is created measuring the qubit in the basis
		specified by the i-th element of a random list of integers
		(a_bases for Alice, b_bases for Bob).

		Parameters
		----------
		a_bases : list[int]
		 			 bases in which Alice performs her measures
		b_bases : list[int]
		 			 bases in which Bob performs his measures
		quantum_states: list[qiskit.QuantumCircuit]
		              list of Bell's states |PSI->

		Returns
		-------
		a_raw_key, b_raw_key : bitstring.Bits, bitstring.Bits
							   the raw keys for Alice and Bob, respectively

		"""

		simulator = Aer.get_backend('aer_simulator')
		a_raw_key = []
		b_raw_key = []

		# generates one bit at time
		for i in range(len(a_bases)):
			circuit = self._make_circuit(a_bases[i], b_bases[i], quantum_states[i])
			bits = simulator.run(transpile(circuit, simulator), shots=1, memory=True).result().get_memory()[0]
			# bits are string like '0 1', '1 1', ecc, where the first bit is relative to Bob's measure
			a_raw_key.append(int(bits.split(' ')[1]))
			b_raw_key.append(int(bits.split(' ')[0]))

		return Bits(a_raw_key), Bits(b_raw_key)

	def _make_circuit(self, a_basis, b_basis, quantum_state):

		# the resulting circuit is the composition 
		# of the circuit to generate the |PSI-> state, 
		# and the circuit to perform the appropriate measurements 
		# according to a_basis and b_basis

		circuit = QuantumCircuit( QuantumRegister(2, name='q'),
									    ClassicalRegister(1, name='a'),
									    ClassicalRegister(1, name='b') )
		circuit.compose(quantum_state, inplace=True)
		circuit.compose(DecodingCircuit(a_basis, b_basis).circuit, inplace=True)
		return circuit

class DecodingCircuit:
	"""Class to create qiskit circuit to perform measurements according to E91 protocol.
	
	The circuit is created to perform mesurements
	according to the given Alice's (a_basis) and Bob's bases (b_basis).
	
	Both a_basis and b_basis have values in {0,1,2}, with the following meaning:

	Alice:
		0 -> Z basis
		1 -> X basis
		2 -> (Z + X) / sqrt(2) basis

	Bob:
		0 -> Z basis
		1 -> (Z - X) / sqrt(2) basis
		2 -> (Z + X) / sqrt(2) basis

	Since with qiskit is possible to measure only spin component in the Z basis
	(that is, in the computational basis), a_basis and b_basis are interpreted as
	values specifying the unitary transformation to apply to the state to obtain measurement results equivalent
	to actually measure the qubit in the given basis:

	Alice:
		0 -> identity operator
		1 -> Hadamard operator
		2 -> rotation of -pi/4 about y axis
	
	Bob:
		0 -> identity operator
		1 -> rotation of pi/4 about y axis
		2 -> rotation of -pi/4 about y axis


	Parameters
	----------
	a_basis : int
	 			 basis in which Alice performs her measure
	b_basis : int
	 			 basis in which Bob performs his measure

	Attributes
	----------
	self.circuit : qiskit.QuantumCircuit
				 generated circuit

	"""

	def __init__(self, a_basis, b_basis):		
		self.circuit = None
		self._make_circuit(a_basis, b_basis)

	def _make_circuit(self, a_basis, b_basis):
		q = QuantumRegister(2, name='q')
		a = ClassicalRegister(1, name='a')
		b = ClassicalRegister(1, name='b')

		self.circuit = QuantumCircuit(q, a, b)
		# add gates to perform the appropriate unitary transformations
		self.circuit.append(self._a_gate(a_basis), (q[0],))
		self.circuit.append(self._b_gate(b_basis), (q[1],))
		self.circuit.barrier()
		self.circuit.measure([q[0],q[1]],[a[0],b[0]])

	def _a_gate(self, a_basis):
		# implements the logic to choose the appropriate gate for Alice
		if a_basis == 0:
			return IGate()
		if a_basis == 1:
			return HGate()
		if a_basis == 2:
			return RYGate(-pi/4)

	def _b_gate(self, b_basis):
		# implements the logic to choose the appropriate gate for Bob
		if b_basis == 0:
			return IGate()
		if b_basis == 1:
			return RYGate(pi/4)
		if b_basis == 2:
			return RYGate(-pi/4)