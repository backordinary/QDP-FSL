# https://github.com/andres3455/-DEUTSCH-Y-DEUTSCH-JOZSA/blob/a76cec7b334d1627c9aa4fdf1460f5cda76d8d78/Tercer%20Deustch.py
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit import Aer
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
print("Andres Rodriguez")
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