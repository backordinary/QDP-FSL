# https://github.com/JohnLins/QuantumFun/blob/9741df6cae1440734ce9a7eb9ac49db4bddddc46/ROTATE/main.py
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from numpy import pi

qreg_q = QuantumRegister(3, 'q')
creg_c = ClassicalRegister(3, 'c')
circuit = QuantumCircuit(qreg_q, creg_c)

circuit.h(qreg_q[0])
circuit.h(qreg_q[1])
circuit.h(qreg_q[2])
circuit.rz(pi/2, qreg_q[0])
circuit.ry(pi/2, qreg_q[1])
circuit.rz(pi/2, qreg_q[2])
circuit.measure(qreg_q[0], creg_c[0])
circuit.measure(qreg_q[1], creg_c[1])
circuit.measure(qreg_q[2], creg_c[2])