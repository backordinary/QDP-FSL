# https://github.com/Alvaradom08/reporte-final-/blob/6c013c8a25d408bf001b10c293608070fa56419c/DeutschJozsa.py
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit import Aer
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

simulator = Aer.get_backend('qasm_simulator')
#CONSTANTE
print("Funcion 1")
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

print("result",counts)

print(circuit)

plot_histogram(counts)

plt.show()

#BALANCEADA
print("Funcion 2")
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

print("result",counts)

print(circuit)

plot_histogram(counts)

plt.show()

#BALANCEADA
print("Funcion 3")
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

print("resut",counts)

print(circuit)

plot_histogram(counts)

plt.show()