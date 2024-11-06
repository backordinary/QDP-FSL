# https://github.com/devsgnr/learning-quantum/blob/475507bc0c5aa55c606da69b55010dc63ef425a2/src/qiskit_02.py
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

circuit = QuantumCircuit(2)
theta = Parameter('θ')

circuit.rz(theta, 0)
circuit.crz(theta, 1, 0)
circuit.draw()
