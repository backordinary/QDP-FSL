# https://github.com/Mateo0laya/Algoritmo-de-Deutsch-y-Deutsch-Josza/blob/b681057e33488054a39a9c0d79d0deaec6642d02/Todos%20a%200.py
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit import Aer
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
# Use Aer's qasm_simulator
simulator = Aer.get_backend('qasm_simulator')

#Funcion todos a 0


#Caso (0,0)
# Create a Quantum Circuit acting on the q register
circuit = QuantumCircuit(2, 2)
# Map the quantum measurement to the classical bits
circuit.measure([0,1], [0,1])
# compile the circuit down to low-level QASM instructions
# supported by the backend (not needed for simple circuits)
compiled_circuit = transpile(circuit, simulator)# Execute the circuit on the qasm simulator
job = simulator.run(compiled_circuit, shots=1000)
# Grab results from the job
result = job.result()
# Returns counts
counts = result.get_counts(circuit)
print("\nTotal count for 00 and 11 are:",counts)
# Draw the circuit
print(circuit)
# Plot a histogram
plot_histogram(counts)
plt.show()

#Caso (0,1)
# Create a Quantum Circuit acting on the q register
circuit = QuantumCircuit(2, 2)
#Add a Not gate on qubit 1
circuit.x(1)
# Map the quantum measurement to the classical bits
circuit.measure([0,1], [0,1])
# compile the circuit down to low-level QASM instructions
# supported by the backend (not needed for simple circuits)
compiled_circuit = transpile(circuit, simulator)# Execute the circuit on the qasm simulator
job = simulator.run(compiled_circuit, shots=1000)
# Grab results from the job
result = job.result()
# Returns counts
counts = result.get_counts(circuit)
print("\nTotal count for 00 and 11 are:",counts)
# Draw the circuit
print(circuit)
# Plot a histogram
plot_histogram(counts)
plt.show()

#Caso (1,0)
# Create a Quantum Circuit acting on the q register
circuit = QuantumCircuit(2, 2)
#Add a Not gate on quibit 0
circuit.x(0)
# Map the quantum measurement to the classical bits
circuit.measure([0,1], [0,1])
# compile the circuit down to low-level QASM instructions
# supported by the backend (not needed for simple circuits)
compiled_circuit = transpile(circuit, simulator)# Execute the circuit on the qasm simulator
job = simulator.run(compiled_circuit, shots=1000)
# Grab results from the job
result = job.result()
# Returns counts
counts = result.get_counts(circuit)
print("\nTotal count for 00 and 11 are:",counts)
# Draw the circuit
print(circuit)
# Plot a histogram
plot_histogram(counts)
plt.show()

#Caso (1,1)
# Create a Quantum Circuit acting on the q register
circuit = QuantumCircuit(2, 2)
#Add a Not gate on quibit 0 and quibit 1
circuit.x(0)
circuit.x(1)
# Map the quantum measurement to the classical bits
circuit.measure([0,1], [0,1])
# compile the circuit down to low-level QASM instructions
# supported by the backend (not needed for simple circuits)
compiled_circuit = transpile(circuit, simulator)# Execute the circuit on the qasm simulator
job = simulator.run(compiled_circuit, shots=1000)
# Grab results from the job
result = job.result()
# Returns counts
counts = result.get_counts(circuit)
print("\nTotal count for 00 and 11 are:",counts)
# Draw the circuit
print(circuit)
# Plot a histogram
plot_histogram(counts)
plt.show()