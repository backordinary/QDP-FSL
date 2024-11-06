# https://github.com/NetwCodeProjects/Quantum-Neural-Network/blob/fbe3e7fe29a34d9004e8dd00b8a0d550df1d9696/deutsch_jozsa_algorithm/deutsch_jozsa_algorithm.py
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from numpy import pi

qreg_q = QuantumRegister(3, 'q')
creg_c = ClassicalRegister(3, 'c')
circuit = QuantumCircuit(qreg_q, creg_c)

circuit.h(qreg_q[1])
circuit.cx(qreg_q[1], qreg_q[2])
circuit.cx(qreg_q[0], qreg_q[1])
circuit.h(qreg_q[0])
circuit.barrier(qreg_q)
circuit.measure(qreg_q[0], creg_c[0])
circuit.measure(qreg_q[1], creg_c[1])
circuit.barrier(qreg_q)
circuit.x(qreg_q[2])
circuit.x(qreg_q[2]).c_if(creg_c, 1)
circuit.z(qreg_q[2])
circuit.z(qreg_q[2]).c_if(creg_c, 2)
circuit.measure(qreg_q[2], creg_c[2])