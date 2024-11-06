# https://github.com/eolocon/qkd_project/blob/ea2e69ccc69d165ebdd539e1a127347203a8cb7f/src/e91/encoding.py
from qiskit import QuantumCircuit

class Encoder:
	"""Encoder for E91 protocol.

	Creates an array filled with the Bell's state |PSI-> = (|01> - |10>) / sqrt(2)

	"""

	def encode(self, length:int):
		"""Creates an array filled with the Bell's state |PSI->

		Parameters
		----------
		length : int
			   length of the array

		Returns
		-------
		list(qiskit.QuantumCircuit)

		"""
		return [EncodingCircuit().circuit for i in range(length)]

class EncodingCircuit:
	""" Qiskit circuit to create the Bell's state |PSI-> 

	The |PSI-> state can be obtained applying the operator ('x' denotes the tensor product)		
	(CNOT)(H x I)(X x X) to the state |00>.
	
	Attributes
	----------
	circuit : qiskit.QuantumCircuit
	        circuit to create the Bell's state

	"""

	def __init__(self):
		self.circuit = None
		self._make_circuit()

	def _make_circuit(self):
		self.circuit = QuantumCircuit(2)
		self.circuit.x(0)
		self.circuit.x(1)
		self.circuit.h(0)
		self.circuit.cnot(0,1)
		self.circuit.barrier()
