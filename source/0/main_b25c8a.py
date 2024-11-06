# https://github.com/Suryansh-23/Quantum-Circuits/blob/f8956d8091f72fd40e9ed5e8e5d2493d44c4d401/Exp%232/main.py

#The code for the circuit to perform on the simulator and the ibm computer is same

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from numpy import pi

qreg_q = QuantumRegister(1, "q")
creg_c = ClassicalRegister(1, "c")
circuit = QuantumCircuit(qreg_q, creg_c)

circuit.y(qreg_q[0])
circuit.measure(qreg_q[0], creg_c[0])
