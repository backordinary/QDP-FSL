# https://github.com/eolocon/qkd_project/blob/5af2d49f58fe6073c3bb5bff00d84b2f7755c171/src/bb84/encoding.py
from qiskit import QuantumCircuit

class Encoder:
	"""Encoder for BB84 protocol.

	Creates an array of states according to the protocol

	"""
	def encode(self, a_raw_key, a_bases):
		"""Creates an array of states according to the protocol

		Parameters
		----------
		a_raw_key : bits.Bitstring
			      bits of Alice's raw key
		a_bases : list[int]
			    bases in which Alice prepares each state

		Returns
		-------
		list(qiskit.QuantumCircuit)

		"""
		return [EncodingCircuit(a_raw_key[i], a_bases[i]).circuit for i in range(len(a_raw_key))]

class EncodingCircuit:
	""" Qiskit circuit to create the B884 states 

	According to the protocol, the states are:

	|0> if a_raw_bit == 0 and a_basis == 0
	|1> if a_raw_bit == 1 and a_basis == 0
	|+> if a_raw_bit == 0 and a_basis == 1
	|-> if a_raw_bit == 1 and a_basis == 1
	
	Attributes
	----------
	circuit : qiskit.QuantumCircuit
	        circuit to create the state

	"""
	def __init__(self, a_raw_bit, a_basis):
		self.circuit = None
		self._make_circuit(a_raw_bit, a_basis)

	def _make_circuit(self, a_raw_bit, a_basis):
		self.circuit = QuantumCircuit(1)
		if a_raw_bit:
			self.circuit.x(0)
		if a_basis:
			self.circuit.h(0)
		self.circuit.barrier()