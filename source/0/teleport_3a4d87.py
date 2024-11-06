# https://github.com/gmgalvan/quantum_teleportation/blob/ebd96f275f14f54ecc94d027d0cebca56aeb62cd/teleport.py
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from numpy import pi

qreg_q = QuantumRegister(5, 'q')
creg_c = ClassicalRegister(5, 'c')
circuit = QuantumCircuit(qreg_q, creg_c)

circuit.h(qreg_q[2])
circuit.cx(qreg_q[2], qreg_q[4])
circuit.barrier(qreg_q[0], qreg_q[1], qreg_q[2], qreg_q[3], qreg_q[4])
circuit.x(qreg_q[0])
circuit.h(qreg_q[0])
circuit.t(qreg_q[0])
circuit.barrier(qreg_q[0], qreg_q[1], qreg_q[2], qreg_q[3], qreg_q[4])
circuit.h(qreg_q[0])
circuit.h(qreg_q[2])
circuit.cx(qreg_q[2], qreg_q[0])
circuit.h(qreg_q[2])
circuit.measure(qreg_q[0], creg_c[0])
circuit.measure(qreg_q[2], creg_c[2])
circuit.barrier(qreg_q[3], qreg_q[4])
circuit.x(qreg_q[4])
circuit.z(qreg_q[4])
circuit.barrier(qreg_q[3], qreg_q[4])
circuit.tdg(qreg_q[4])
circuit.h(qreg_q[4])
circuit.x(qreg_q[4])
circuit.measure(qreg_q[4], creg_c[4])