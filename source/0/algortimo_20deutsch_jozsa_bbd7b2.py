# https://github.com/Naetffy/CNYT-proyecto/blob/8cb14c29abf2b73f4ffa002d05ae9b76b216e50e/Algortimo%20Deutsch-jozsa.py
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit import Aer
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

simulator = Aer.get_backend('qasm_simulator')
##FUNCION CONSTANTE##
print("Funcion 1")
print("0000--->0")
print("0001--->0")
print("0010--->0")
print("0011--->0")
print("0100--->0")
print("0101--->0")
print("0110--->0")
print("0111--->0")
print("1000--->0")
print("1001--->0")
print("1010--->0")
print("1011--->0")
print("1100--->0")
print("1101--->0")
print("1110--->0")
print("1111--->0")
circuit = QuantumCircuit(5, 4)

circuit.x(4)
circuit.barrier()
circuit.h(0)
circuit.h(1)
circuit.h(2)
circuit.h(3)
circuit.h(4)
circuit.barrier()

circuit.barrier()
circuit.h(0)
circuit.h(1)
circuit.h(2)
circuit.h(3)
circuit.barrier()
circuit.measure([0,1,2,3],[3,2,1,0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

print("\nTotal count for 0000 and 1111 are:",counts)

print(circuit)

plot_histogram(counts)

plt.show()

##FUNCION BALANCEADA##
print("Funcion 2")
print("0000--->0")
print("0001--->0")
print("0010--->0")
print("0011--->0")
print("0100--->0")
print("0101--->0")
print("0110--->0")
print("0111--->0")
print("1000--->1")
print("1001--->1")
print("1010--->1")
print("1011--->1")
print("1100--->1")
print("1101--->1")
print("1110--->1")
print("1111--->1")
circuit = QuantumCircuit(5, 4)

circuit.x(4)
circuit.barrier()
circuit.h(0)
circuit.h(1)
circuit.h(2)
circuit.h(3)
circuit.h(4)
circuit.barrier()
circuit.cnot(0,4)
circuit.barrier()
circuit.h(0)
circuit.h(1)
circuit.h(2)
circuit.h(3)
circuit.barrier()
circuit.measure([0,1,2,3],[3,2,1,0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

print("\nTotal count for 0000 and 1111 are:",counts)

print(circuit)

plot_histogram(counts)

plt.show()

##FUNCION BALANCEADA##
print("Funcion 3")
print("0000--->0")
print("0001--->0")
print("0010--->0")
print("0011--->0")
print("0100--->1")
print("0101--->1")
print("0110--->1")
print("0111--->1")
print("1000--->0")
print("1001--->0")
print("1010--->0")
print("1011--->0")
print("1100--->1")
print("1101--->1")
print("1110--->1")
print("1111--->1")
circuit = QuantumCircuit(5, 4)

circuit.x(4)
circuit.barrier()
circuit.h(0)
circuit.h(1)
circuit.h(2)
circuit.h(3)
circuit.h(4)
circuit.barrier()
circuit.cnot(1,4)
circuit.barrier()
circuit.h(0)
circuit.h(1)
circuit.h(2)
circuit.h(3)
circuit.barrier()
circuit.measure([0,1,2,3],[3,2,1,0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

print("\nTotal count for 0000 and 1111 are:",counts)

print(circuit)

plot_histogram(counts)

plt.show()

##FUNCION BALANCEADA##
print("Funcion 4")
print("0000--->0")
print("0001--->0")
print("0010--->1")
print("0011--->1")
print("0100--->0")
print("0101--->0")
print("0110--->1")
print("0111--->1")
print("1000--->0")
print("1001--->0")
print("1010--->1")
print("1011--->1")
print("1100--->0")
print("1101--->0")
print("1110--->1")
print("1111--->1")
circuit = QuantumCircuit(5, 4)

circuit.x(4)
circuit.barrier()
circuit.h(0)
circuit.h(1)
circuit.h(2)
circuit.h(3)
circuit.h(4)
circuit.barrier()
circuit.cnot(2,4)
circuit.barrier()
circuit.h(0)
circuit.h(1)
circuit.h(2)
circuit.h(3)
circuit.barrier()
circuit.measure([0,1,2,3],[3,2,1,0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

print("\nTotal count for 0000 and 1111 are:",counts)

print(circuit)

plot_histogram(counts)

plt.show()