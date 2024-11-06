# https://github.com/Naetffy/CNYT-proyecto/blob/8cb14c29abf2b73f4ffa002d05ae9b76b216e50e/Algoritmo%20Deutsch.py
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit import Aer
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

simulator = Aer.get_backend('qasm_simulator')

print("Funcion 1")
print("0--->1")
print("1--->1")
circuit = QuantumCircuit(2, 1)

circuit.x(1)
circuit.barrier()
circuit.h(0)
circuit.h(1)
circuit.barrier()
circuit.x(1)
circuit.barrier()
circuit.h(0)
circuit.barrier()
circuit.measure([0],[0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

print("\nTotal count for 0 and 1 are:",counts)

print(circuit)

plot_histogram(counts)

plt.show()

print("Funcion 2")
print("0--->0")
print("1--->1")
circuit = QuantumCircuit(2, 1)

circuit.x(1)
circuit.barrier()
circuit.h(0)
circuit.h(1)
circuit.barrier()
circuit.cnot(0,1)
circuit.barrier()
circuit.h(0)
circuit.barrier()
circuit.measure([0],[0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

print("\nTotal count for 0 and 1 are:",counts)

print(circuit)

plot_histogram(counts)

plt.show()

print("Funcion 3")
print("0--->1")
print("1--->0")
circuit = QuantumCircuit(2, 1)

circuit.x(1)
circuit.barrier()
circuit.h(0)
circuit.h(1)
circuit.barrier()
circuit.x(0)
circuit.cnot(0,1)
circuit.x(0)
circuit.barrier()
circuit.h(0)
circuit.barrier()
circuit.measure([0],[0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

print("\nTotal count for 0 and 1 are:",counts)

print(circuit)

plot_histogram(counts)

plt.show()

print("Funcion 4")
print("0--->0")
print("1--->0")
circuit = QuantumCircuit(2, 1)

circuit.x(1)
circuit.barrier()
circuit.h(0)
circuit.h(1)
circuit.barrier()

circuit.barrier()
circuit.h(0)
circuit.barrier()
circuit.measure([0],[0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

print("\nTotal count for 0 and 1 are:",counts)

print(circuit)

plot_histogram(counts)

plt.show()