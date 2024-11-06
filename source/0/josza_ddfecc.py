# https://github.com/andres3455/-DEUTSCH-Y-DEUTSCH-JOZSA/blob/f3b07d7122f3175b2b4103eca591c6ece6f386bc/JOSZA.py
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit import Aer
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

simulator = Aer.get_backend('qasm_simulator')
print("Funcion 1")
#("0000--->0")
#("0001--->0")
#("0010--->0")
#("0011--->0")
#("0100--->0")
#("0101--->0")
#("0110--->0")
#("0111--->0")
#("1000--->0")
#("1001--->0")
#("1010--->0")
#("1011--->0")
#("1100--->0")
#("1101--->0")
#("1110--->0")
#("1111--->0")
print("ANDRES RODRIGUEZ")
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

print("Funcion 2")
print("ANDRES RODRIGUEZ")
#("0000--->0")
#("0001--->0")
#("0010--->0")
#("0011--->0")
#("0100--->1")
#("0101--->1")
#("0110--->1")
#("0111--->1")
#("1000--->0")
#("1001--->0")
#("1010--->0")
#("1011--->0")
#("1100--->1")
#("1101--->1")
#("1110--->1")
#("1111--->1")
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

print("Funcion 3")
print("ANDRES RODRIGUEZ")
#("0000--->0")
#("0001--->0")
#("0010--->1")
#("0011--->1")
#("0100--->0")
#("0101--->0")
#("0110--->1")
#("0111--->1")
#("1000--->0")
#("1001--->0")
#("1010--->1")
#("1011--->1")
#("1100--->0")
#("1101--->0")
#("1110--->1")
#("1111--->1")
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