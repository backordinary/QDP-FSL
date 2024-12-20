# https://github.com/Hazunanafaru/qiskit-starter/blob/a88e9cea6d654d2e8ec827bd37c1a90ec9d5e40e/0-Introduction/main.py
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.providers.aer import QasmSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
from qiskit import IBMQ

# Load API Key from .env file
from decouple import config
IBMQ.save_account(config('IBM_TOKEN'))


# Set AER Qasm Simulator
simulator = QasmSimulator()

# Create a Quantum Circuit acting on the q register
circuit = QuantumCircuit(2, 2)

# Add a H gate on qubit 0
circuit.h(0)

# Add a CX (CNOT) gate on control qubit 0 and target qubit 1
circuit.cx(0, 1)

# Map the quantum measurement to the classical bits
circuit.measure([0,1], [0,1])

# compile the circuit down to low-level QASM instructions
# supported by the backend (not needed for simple circuits)
compiled_circuit = transpile(circuit, simulator)

# Execute the circuit on the qasm simulator
job = simulator.run(compiled_circuit, shots=1000)

# Grab results from the job
result = job.result()

# Returns counts
counts = result.get_counts(compiled_circuit)
print("\nTotal count for 00 and 11 are:",counts)

# Draw the circuit
circuit.draw(output='mpl', filename='circuit.png')

# Plot Hsitogram
plot_histogram(counts)
plt.savefig('histogram.png')