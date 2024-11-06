# https://github.com/Top-Gun-Maxverick/TestingQCNN/blob/e80e8c962749fbcb01c023eb6944e3c56ad9546a/sample1.py
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from numpy import pi

qreg_q = QuantumRegister(4, 'q')
creg_c = ClassicalRegister(4, 'c')
circuit = QuantumCircuit(qreg_q, creg_c)

circuit.h(qreg_q[0])
circuit.cx(qreg_q[0], qreg_q[1])
circuit.barrier(qreg_q[0])
circuit.cx(qreg_q[1], qreg_q[2])
circuit.barrier(qreg_q[0])
circuit.barrier(qreg_q[1])
circuit.cx(qreg_q[2], qreg_q[3])
circuit.u(pi/2, pi/2, pi/2, qreg_q[0])
circuit.u(pi/2, pi/2, pi/2, qreg_q[1])
circuit.u(pi/2, pi/2, pi/2, qreg_q[2])
circuit.u(pi/2, pi/2, pi/2, qreg_q[3])
circuit.barrier(qreg_q[0])
circuit.barrier(qreg_q[1])
circuit.cx(qreg_q[2], qreg_q[3])
circuit.cx(qreg_q[1], qreg_q[2])
circuit.barrier(qreg_q[0])
circuit.cx(qreg_q[0], qreg_q[1])
circuit.h(qreg_q[0])
circuit.measure(qreg_q[0], creg_c[0])
circuit.measure(qreg_q[1], creg_c[1])
circuit.measure(qreg_q[2], creg_c[2])
circuit.measure(qreg_q[3], creg_c[3])
circuit.draw('mpl')

#editor = CircuitComposer(circuit=circuit)
#editor
