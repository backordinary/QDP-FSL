# https://github.com/guilt/Qubit-Course/blob/bba9053e893162a32af5e81aabf9a74388dca1fa/02-Lecture/00-qiskit-start.py
#!/usr/bin/env python
import numpy as np
from qiskit import Aer, QuantumCircuit, execute

# Use Aer's qasm_simulator
simulator = Aer.get_backend('qasm_simulator')

# Create a Quantum Circuit acting on the q register
circuit = QuantumCircuit(2, 2)

circuit.h(0)
circuit.cx(0, 1)
circuit.measure([0,1], [0,1])

# Draw the circuit
print(circuit.draw())

# Execute the circuit on the Simulator
job = execute(circuit, simulator, shots=1000)

# Grab results from the job
result = job.result()
counts = result.get_counts(circuit)
print("\nTotal counts:",counts)
