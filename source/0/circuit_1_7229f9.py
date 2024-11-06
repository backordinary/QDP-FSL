# https://github.com/aash-gates/Quantum-Computing-Qiskit/blob/ff5cf45127f927b2c0c328eadb92ea110545cd34/circuit(1).py
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from numpy import pi

qreg_q = QuantumRegister(3, 'q')
creg_c = ClassicalRegister(3, 'c')
circuit = QuantumCircuit(qreg_q, creg_c)

circuit.measure(qreg_q[0], creg_c[0])
circuit.measure(qreg_q[1], creg_c[1])
circuit.ccx(qreg_q[0], qreg_q[1], qreg_q[2])
circuit.measure(qreg_q[2], creg_c[2])