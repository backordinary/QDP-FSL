# https://github.com/NetwCodeProjects/Quantum-Neural-Network/blob/0957133914183c19440022490310f9e57a8b1b09/quantum_neuralnetwork_circuit/quantum_neural_network_4_qubit.py
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from numpy import pi

qreg_q = QuantumRegister(4, 'q')
creg_c = ClassicalRegister(4, 'c')
circuit = QuantumCircuit(qreg_q, creg_c)

circuit.h(qreg_q[0])
circuit.cx(qreg_q[0], qreg_q[1])
circuit.cx(qreg_q[1], qreg_q[2])
circuit.h(qreg_q[3])
circuit.barrier(qreg_q)
circuit.measure(qreg_q[0], creg_c[0])
circuit.measure(qreg_q[1], creg_c[1])
circuit.measure(qreg_q[2], creg_c[2])
circuit.measure(qreg_q[3], creg_c[3])
# Use the classical measurement results to update the weights of the quantum neural network