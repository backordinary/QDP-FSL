# https://github.com/Robinbux/AI-Projects/blob/edde9e02ad21263ad21ad0d3bff64e78c556d587/QKNN/qknn.py
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from numpy import pi

qreg_q = QuantumRegister(3, 'q')
creg_c = ClassicalRegister(3, 'c')
circuit = QuantumCircuit(qreg_q, creg_c)

circuit.h(qreg_q[0])
circuit.ccx(qreg_q[0], qreg_q[1], qreg_q[2])
circuit.h(qreg_q[0])
circuit.measure(qreg_q[0], creg_c[0])
print("TEST")
circuit.draw(output="mpl")