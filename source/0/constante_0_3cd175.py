# https://github.com/Nat15005/DeutschYDeutsch-Jozsa/blob/07c15bcc325bfc04979230769f4075e1124fb0b9/Deutsch/Constante_0.py
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit import Aer
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
# Use Aer's qasm_simulator
simulator = Aer.get_backend('qasm_simulator')
# Create a Quantum Circuit acting on the q register
circuit = QuantumCircuit(2, 1)
# Add a H gate on qubit 0


# Add a CX (CNOT) gate on control qubit 0 and target qubit 1

circuit.x(1)
circuit.barrier(0,1)
circuit.h(0)
circuit.h(1)
circuit.barrier(0,1)
circuit.id(0)
circuit.id(1)
circuit.barrier(0,1)
circuit.h(0)
circuit.barrier(0,1)

# Map the quantum measurement to the classical bits
circuit.measure([0], [0])
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