# https://github.com/dcvdiego/Quantum/blob/efede2222bf1439cb13f8e6f5395d78573160a35/first_circuit.py
from qiskit import *

circuit = QuantumCircuit(2, 2)

# quantum_register = QuantumRegister(2)
# classical_register = ClassicalRegister(2)
# circuit = QuantumCircuit(quantum_register,classical_register)


circuit.draw(output='mpl')
