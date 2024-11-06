# https://github.com/eolocon/qkd_project/blob/839380c126181df84b5a13c38d77bd1852d5c470/src/bb84/decoding.py
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, transpile
from bitstring import Bits

class Decoder:
	"""Decoder for BB84 protocol.

	Generates Bob's raw key, using the quantum states prepared by Alice and Bob's chosen bases.

	"""

	def decode(self,b_bases, quantum_states):
		"""Generates Bob's raw key. 

		The i-th bit of the key is created measuring the qubit in the basis
		specified by the i-th element of a random bit string b_bases

		Parameters
		----------
		b_bases : list[int]
		 	    bases in which Bob performs his measures
		quantum_states: list[qiskit.QuantumCircuit]
		              list of states prepared by Alice

		Returns
		-------
		b_raw_key : bitstring.Bits
				  Bob's raw key

		"""

		b_raw_key = []

		simulator = Aer.get_backend('aer_simulator')

		# generates one bit at time
		for i in range(len(b_bases)):
			circuit = self._make_circuit(b_bases[i], quantum_states[i])
			bit = simulator.run(transpile(circuit, simulator), shots=1, memory=True).result().get_memory()[0]
			# bit is the string '0' or '1'
			b_raw_key.append(int(bit))

		return Bits(b_raw_key)

	def _make_circuit(self, b_basis, quantum_state):

		# the resulting circuit is the composition 
		# of the circuit assembled by Alice, 
		# and the circuit to perform the appropriate measurements 
		# according b_basis

		circuit = QuantumCircuit( QuantumRegister(1, name='q'),
								  ClassicalRegister(1, name='raw_key_bit') )
		circuit.compose(quantum_state, inplace=True)
		circuit.compose(DecodingCircuit(b_basis).circuit, inplace=True)
		return circuit

class DecodingCircuit:
	"""Class to create qiskit circuit to perform measurements according to BB84 protocol.
	
	The circuit is created to perform mesurements Bob's chosen basis (b_basis). 
	
	b_basis has values in {0,1}, with the following meaning:

		0 -> Z basis
		1 -> X basis

	Since with qiskit is possible to measure only spin component in the Z basis
	(that is, in the computational basis), b_basis is interpreted as
	the value specifying the unitary transformation to apply to the state to obtain measurement results equivalent
	to actually measure the qubit in the given basis.

	In this case, if b_basis = 1, the Hadamard operator is applied.

	Parameters
	----------
	b_basis : int
	 			 basis in which Bob performs his measure

	Attributes
	----------
	self.circuit : qiskit.QuantumCircuit
				 generated circuit

	"""

	def __init__(self, b_basis):		
		self.circuit = None
		self._make_circuit(b_basis)

	def _make_circuit(self, b_basis):
		q = QuantumRegister(1, name='q')
		key_bit = ClassicalRegister(1, name='key_bit')

		self.circuit = QuantumCircuit(q, key_bit)
		if b_basis:
			self.circuit.h(0)
		self.circuit.barrier()
		self.circuit.measure([q[0]],[key_bit[0]])