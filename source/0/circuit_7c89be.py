# https://github.com/CCNYseniors/rubik/blob/92ea9cbe816d85db447b8c6cdedf42b4e3f4e7c1/circuit.py
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from numpy import pi

qreg_q = QuantumRegister(2, 'q')
creg_c = ClassicalRegister(2, 'c')
circuit = QuantumCircuit(qreg_q, creg_c)

circuit.x(qreg_q[0])
circuit.x(qreg_q[0])
circuit.x(qreg_q[0])
circuit.h(qreg_q[0])
circuit.cx(qreg_q[0], qreg_q[1])
circuit.measure(qreg_q[0], creg_c[0])
circuit.rx(pi/3, qreg_q[0])
circuit.measure(qreg_q[1], creg_c[1])
circuit.ry(pi/5, qreg_q[1])