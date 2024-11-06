# https://github.com/quantumtrace/quantum-coins/blob/187133ff115d44be2bf1816c5abb5eeebc99fe70/quantum_coins_ibmq_belem.py
#Circuit executed by the ibmq_belem device
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from numpy import pi

qreg_q = QuantumRegister(5, 'q')
creg_c = ClassicalRegister(3, 'c')
circuit = QuantumCircuit(qreg_q, creg_c)

circuit.rz(1.5707963267948961, qreg_q[0])
circuit.sx(qreg_q[0])
circuit.rz(1.5707963267948966, qreg_q[0])
circuit.rz(1.5707963267948961, qreg_q[1])
circuit.sx(qreg_q[1])
circuit.rz(1.5707963267948966, qreg_q[1])
circuit.rz(1.5707963267948961, qreg_q[2])
circuit.sx(qreg_q[2])
circuit.rz(1.5707963267948966, qreg_q[2])
circuit.measure(qreg_q[0], creg_c[0])
circuit.measure(qreg_q[1], creg_c[1])
circuit.measure(qreg_q[2], creg_c[2])