# https://github.com/commanderHR1/commanderHR1/blob/d16c63268c2c7d554ae5a30d5e96ce163880db77/qiskit/firstStartExample.py
# Example program from QisKit Getting Started Guide
# Typical program flow:
# - Build: Design a quantum circuit according to the problem
# - Execute: Run experiments on multiple backends
# - Analyze: Calculate and visualize results

import numpy as np
from qiskit import(
  QuantumCircuit,
  execute,
  Aer)
from qiskit.visualization import plot_histogram

# Use Aer's qasm_simulator (classical simulation of a quantum computer)
simulator = Aer.get_backend('qasm_simulator')

# Create a Quantum Circuit acting on the q register
circuit = QuantumCircuit(2, 2)

# Add a H gate on qubit 0
circuit.h(0)

# Add a CX (CNOT) gate on control qubit 0 and target qubit 1
circuit.cx(0, 1)

# Map the quantum measurement to the classical bits
circuit.measure([0,1], [0,1])

# Execute the circuit on the qasm simulator
job = execute(circuit, simulator, shots=1000)

# Grab results from the job
result = job.result()

# Returns counts
counts = result.get_counts(circuit)
print("\nTotal count for 00 and 11 are:",counts)

# Draw the circuit
circuit.draw()
