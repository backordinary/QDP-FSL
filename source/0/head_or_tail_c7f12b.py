# https://github.com/ccicconetti/qiskit-tests/blob/8e362a1d041b071b64b9630201e0bb56246c735a/utils/head_or_tail.py
import numpy as np
import matplotlib.pyplot as plt
from qiskit import(
  QuantumCircuit,
  execute,
  Aer)
from qiskit.visualization import plot_histogram

# Use Aer's qasm_simulator
simulator = Aer.get_backend('qasm_simulator')

# Create a Quantum Circuit acting on the q register
circuit = QuantumCircuit(1, 1)

# Add a H gate on qubit 0
circuit.h(0)

# Map the quantum measurement to the classical bits
circuit.measure([0], [0])

# Execute the circuit on the qasm simulator
job = execute(circuit, simulator, shots=1)

# Grab results from the job
result = job.result()

# Returns counts
counts = result.get_counts(circuit)

assert '0' in counts or '1' in counts

if '0' in counts:
    assert counts['0'] == 1
    print("head")
else:
    assert '1' in counts
    assert counts['1'] == 1
    print("tail")
