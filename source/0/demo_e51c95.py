# https://github.com/pradeepchauhan7/Quantum/blob/2694ceefd23b884899b656eb555adc5853d7a304/demo.py
#%%
import numpy as np
from qiskit import(
    QuantumCircuit,
    execute,
    Aer
)

from qiskit.visualization import plot_histogram

# Use Aer's qasm_simulator
simulator = Aer.get_backend('qasm_simulator')

# Create a Quantum Circuit acting on the q register
circuit = QuantumCircuit(2, 2)

# Add a H gate on qubit 0
circuit.h(0)

# Add a CX(CNOT) gate on control qubit 0 and target qubit 1
circuit.cx(0, 1)

# Map the Quantum measurement to the classical bits
circuit.measure([0,1], [0,1])

# Execute the circuit on the qasm simulator
job = execute(circuit, simulator, shots=1000)

# Grab results from the job
result = job.result()

# Returns count
counts = result.get_counts(circuit)
print("\nTotal count for 00 and 11 are:", counts)

# Draw the circuit
circuit.draw()

# plot_histogram(counts)
# %%
