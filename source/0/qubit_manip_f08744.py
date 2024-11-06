# https://github.com/dennisbiber/quantium/blob/de4e95eff2035035bea3717ebc9a26784929fd40/qubit_manip.py
import pennylane as qml
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from pennylane import numpy as np

dev1 = qml.device("default.qubit",
				 wires=2,
				 shots=5000)

dev2 = qml.device("default.qubit",
				 wires="wire1",
				 shots=1000)

dev3 = qml.device("default.qubit",
				 wires="wire2",
				 shots=1000)

@qml.qnode(dev1)
def circuit0(phi, theta):
	qml.RX(phi[0], wires=0)
	qml.RY(phi[1], wires=1)
	qml.CNOT(wires=[0, 1])
	qml.PhaseShift(theta, wires=0)
	return qml.expval(qml.PauliZ(0))

phi = np.array([.5,.2,.4])
theta = .2
dcircuit = qml.grad(circuit0)
dcircuit(phi, theta)
print(dir(dcircuit))
print(dcircuit._grad_with_forward)

@qml.qnode(dev2)
def ciruit1(params, theta):
	qml.RX(params[0], wires=0)
	qml.CNOT(wires=[0, 1])
	qml.PhaseShift(theta, wires=0)
	return qml.expval(qml._PauliZ(0))

phi = np.array([.8,.15,.5])
theta = .1
dcircuit = qml.grad(circuit1)
dcircuit(phi, theta)
print(dir(dcircuit))
print(dcircuit._grad_with_forward)

@qml.qnode(dev3)
def ciruit2(params, theta):
	qml.RX(params[0], wires=0)
	qml.CNOT(wires=[0, 1])
	qml.PhaseShift(theta, wires=0)
	return qml.expval(qml._PauliZ(0))

phi = np.array([.95,.05,.25])
theta = .2
dcircuit = qml.grad(circuit0)
dcircuit(phi, theta)
print(dir(dcircuit))
print(dcircuit._grad_with_forward)