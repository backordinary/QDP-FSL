# https://github.com/aash-gates/Quantum-Computing-Qiskit/blob/ff5cf45127f927b2c0c328eadb92ea110545cd34/Binary%20Counter.py
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from numpy import pi

qreg_q = QuantumRegister(5, 'q')
creg_c = ClassicalRegister(5, 'c')
circuit = QuantumCircuit(qreg_q, creg_c)

circuit.rx(pi/2, qreg_q[0])
circuit.z(qreg_q[1])
circuit.h(qreg_q[2])
circuit.ry(pi/2, qreg_q[3])
circuit.measure(qreg_q[4], creg_c[4])
circuit.y(qreg_q[0])
circuit.rx(pi/2, qreg_q[1])
circuit.rx(pi/2, qreg_q[3])
circuit.y(qreg_q[0])
circuit.swap(qreg_q[1], qreg_q[3])
circuit.z(qreg_q[1])
circuit.measure(qreg_q[3], creg_c[3])
circuit.cx(qreg_q[2], qreg_q[4])
circuit.measure(qreg_q[1], creg_c[1])
circuit.rz(pi/2, qreg_q[2])
circuit.rx(pi/2, qreg_q[4])
circuit.ry(pi/2, qreg_q[4])