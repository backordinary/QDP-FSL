# https://github.com/ASU-KE/QuantumCollaborativeSamples/blob/239d8527cb3eabcb5d5d66593ad4d1d7db1ba1e4/3-simple_circuit.py
# Modified to specify the Hub/Group/Project.
# From: https://qiskit.org/documentation/intro_tutorial1.html

import numpy as np
from qiskit import IBMQ
from qiskit import QuantumCircuit, transpile
from qiskit.providers.aer import QasmSimulator
from qiskit.visualization import plot_histogram

# This uses credentials that have been saved to disk using 1-save_credentials.py
IBMQ.load_account()

# Select the ASU Quantum Hub Provider (Hub, Group, and Project) - change PROJECT to your assigned project
provider = IBMQ.get_provider(hub="ibm-q-asu", group="main", project="PROJECT")
 
# Use Aer's qasm_simulator
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
circuit.draw()
