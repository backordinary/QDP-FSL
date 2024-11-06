# https://github.com/sujitmandal/Quantum-Programming/blob/1d348c307584bb804255eab75357789786ac5437/ConstructEmptyQuantumCircuit.py
from qiskit import Aer
from qiskit import execute
from qiskit import QuantumCircuit
from qiskit import QuantumRegister
from qiskit import ClassicalRegister

'''
This programe is create by Sujit Mandal
Github: https://github.com/sujitmandal
Pypi : https://pypi.org/user/sujitmandal/
LinkedIn : https://www.linkedin.com/in/sujit-mandal-91215013a/
'''

circuit = QuantumCircuit(2)
circuit.h(0)
circuit.cx(0, 1)

simulator = Aer.get_backend('unitary_simulator')

result = execute(circuit, simulator).result()
unitary = result.get_unitary(circuit)
print("Circuit unitary:\n", unitary)