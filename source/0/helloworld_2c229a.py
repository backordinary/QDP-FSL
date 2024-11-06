# https://github.com/TheArctesian/QuantumStuff/blob/6e01b01fa2a0b0969e92d75d3f011ecbf22b820c/Test/helloworld.py
from qiskit import *

qr = qiskit.QuantumRegister(2) # 2 qubits register
cr = qiskit.ClassicalRegister(2) # 2 classical bits register
circuit = qiskit.QuantumCircuit(qr, cr) # quantum circuit
# %matplotlib inline
# qc.draw()
# this only works in jupyter notebook
circuit.h(qr[0]) # Hadamard gate
circuit.cx(qr[0], qr[1]) # CNOT
circuit.measure(qr, cr) # measure

print(circuit)