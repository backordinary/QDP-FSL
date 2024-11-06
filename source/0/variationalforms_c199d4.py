# https://github.com/Phystro/VQEProject/blob/0fb207a5f0a15e9a9931aea0e53448801aa70d96/VariationalForms.py
import numpy as np
from qiskit import QuantumRegister, QuantumCircuit
# from qiskit.circuit import Instruction

class VariationalForms:

	def R(self, theta: float, phi: float):# -> Instruction:
		# num of num_qubits
		num_qubits: int = 1

		# quantum circuit
		qr: QuantumRegister = QuantumRegister(num_qubits)
		rgate: QuantumCircuit = QuantumCircuit(qr, name="$R(\\theta, \phi)$")
		
		# setup the gate
		rgate.ry(theta, qr[0])
		rgate.rz(phi, qr[0])
		return rgate.to_instruction()

	
	def Rd(self, theta: float, phi: float):# -> Instruction:
		# num of num_qubits
		num_qubits = 1
		
		# quantum circuit
		qr = QuantumRegister(num_qubits)
		rdgate = QuantumCircuit(qr, name="$R(\\theta, \phi)^\dagger$")
		
		# setup the gate
		rdgate.rz(phi, qr[0]).inverse()
		rdgate.ry(theta, qr[0]).inverse()
		return rdgate.to_instruction()

	
	def A(self, theta: float, phi: float):# -> Instruction:
		# num of num_qubits
		num_qubits = 2
		
		# quantum circuit
		qr = QuantumRegister(num_qubits)
		agate = QuantumCircuit(qr, name="$A(\\theta, \phi)$")
		
		# Rotation gates
		RGate = self.R(theta, phi)
		RDGate = self.Rd(theta, phi)
		# RGate = r.to_instruction()
		# RDGate = rd.to_instruction()
		
		# Gate setup
		agate.cx(1, 0)
		agate.append(RDGate, [qr[1]])
		agate.cx(0, 1)
		agate.append(RGate, [qr[1]])
		agate.cx(1, 0)
		return agate.to_instruction()


	def draw(self, g, theme: str = "mpl", decomposed: bool = False):
		num_qubits: int = g.num_qubits
		qc = QuantumCircuit(num_qubits)
		qc.append( g, list(np.arange(num_qubits)) )
		if decomposed:
			return qc.decompose().draw(theme)
		return qc.draw("mpl", initial_state=True)

	def decompose(self, g, theme: str = "mpl"):
		num_qubits: int = g.num_qubits
		qc = QuantumCircuit(num_qubits)
		qc.append( g, list(np.arange(num_qubits)) )
		return qc.decompose().draw(theme, initial_state=True)