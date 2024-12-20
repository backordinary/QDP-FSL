# https://github.com/aash-gates/Quantum-Computing-Qiskit/blob/ff5cf45127f927b2c0c328eadb92ea110545cd34/circuit.py
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from numpy import pi

qreg_q = QuantumRegister(3, 'q')
creg_c = ClassicalRegister(3, 'c')
circuit = QuantumCircuit(qreg_q, creg_c)

circuit.h(qreg_q[0])
circuit.z(qreg_q[2])