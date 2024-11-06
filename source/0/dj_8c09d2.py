# https://github.com/PierreEngelstein/MasterRecherche/blob/8e38d1e8825ec0778059867214c141fc04600860/Programmation/DeutschJozsa/dj.py
import numpy as np
from qiskit import *
import matplotlib.pyplot as plt
from qiskit.visualization import latex as _latex

nb_qubits = 3
circuit = QuantumCircuit(nb_qubits, nb_qubits) # Build the quantum circuit

for i in range(nb_qubits):
    circuit.h(i)

circuit.barrier()

circuit.z(0)
circuit.z(2)

circuit.barrier()
for i in range(nb_qubits):
    circuit.h(i)

circuit.barrier()
for i in range(nb_qubits):
    circuit.measure(i, i)

result = qiskit.visualization.circuit_drawer(circuit, output="text")
print(result)

backend = BasicAer.get_backend('qasm_simulator')
shots = 1024

result = qiskit.execute(circuit, backend,shots=shots).result()

if list(result.get_counts(circuit).keys())[0] == '000':
    print("Constant")
else:
    print("Balanced")

print(list(result.get_counts(circuit).keys())[0])
