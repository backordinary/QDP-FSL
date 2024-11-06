# https://github.com/crxt1234/deutsh-y-deutsh-jozsa/blob/97a78a45acebe96cc188899f4449cbeb5a0fa61a/deutsh.py
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit import Aer
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt



# Use Aer's qasm_simulator
simulator = Aer.get_backend('qasm_simulator')
# Create a Quantum Circuit acting on the q register
circuit = QuantumCircuit(2, 2)
#circuit.x(1) [01]
#circuit.x(0) [10]
#->
#circuit.x(0). circuit.x [11]
circuit.barrier(0,1)
## (0,1) -> (1,0)
circuit.x(0)
circuit.cx(0,1)
circuit.x(0)
circuit.barrier(0,1)
# Map the quantum measurement to the classical bits
circuit.measure([0,1], [1,0])
# compile the circuit down to low-level QASM instructions
# supported by the backend (not needed for simple circuits)
compiled_circuit = transpile(circuit, simulator)
# Execute the circuit on the qasm simulator
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



# Use Aer's qasm_simulator
simulator = Aer.get_backend('qasm_simulator')
# Create a Quantum Circuit acting on the q register
circuit = QuantumCircuit(2, 2)
#circuit.x(1) [01]
#circuit.x(0) [10]
#->
#circuit.x(0) circuit.x [11]
circuit.barrier(0,1)
## (0,1) -> (1,0)
circuit.i(0)
circuit.i(1)
#circuit.x(0)
circuit.barrier(0,1)
# Map the quantum measurement to the classical bits
circuit.measure([1,0], [0,1])
# compile the circuit down to low-level QASM instructions
# supported by the backend (not needed for simple circuits)
compiled_circuit = transpile(circuit, simulator)
# Execute the circuit on the qasm simulator
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



# Use Aer's qasm_simulator
simulator = Aer.get_backend('qasm_simulator')
# Create a Quantum Circuit acting on the q register
circuit = QuantumCircuit(2, 2)
#circuit.x(1) [01]
#circuit.x(0) [10]
#->
#circuit.x(0) circuit.x [11]
circuit.barrier(0,1)
## (0,1) -> (0,|)
circuit.cx(0,1)
#circuit.i(1)
#circuit.x(0)
circuit.barrier(0,1)
# Map the quantum measurement to the classical bits
circuit.measure([1,0], [0,1])
# compile the circuit down to low-level QASM instructions
# supported by the backend (not needed for simple circuits)
compiled_circuit = transpile(circuit, simulator)
# Execute the circuit on the qasm simulator
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


simulator = Aer.get_backend('qasm_simulator')
# Create a Quantum Circuit acting on the q register
circuit = QuantumCircuit(2, 2)
#circuit.x(1) [01]
#circuit.x(0) [10]
#->
#circuit.x(0) cicuit.x [11]
circuit.barrier(0,1)
## (0,1) -> (1,1)
circuit.x(1)
circuit.barrier(0,1)
# Map the quantum measurement to the classical bits
circuit.measure([1,0], [0,1])
# compile the circuit down to low-level QASM instructions
# supported by the backend (not needed for simple circuits)
compiled_circuit = transpile(circuit, simulator)
# Execute the circuit on the qasm simulator
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