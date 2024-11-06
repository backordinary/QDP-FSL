# https://github.com/michaeljohncalvey/Q/blob/f443811e718f4eeff89ab97e07f283130eba8db1/first_steps.py
import numpy as np
from matplotlib import pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit.providers.aer import QasmSimulator
from qiskit.visualization import plot_histogram

simulator = QasmSimulator()

# Create quantum circuit with 2 qubits, 1 traditional bit
circuit = QuantumCircuit(2, 1)

# Applies Hadamard basis to qubit 0, putting it into superposition state
circuit.h(0)

# CNOT entangles control qubit 0 with target qubit 1
circuit.cx(0, 1)

# Measuring both qubits and saving ith measurement in ith classical bit.
circuit.measure([1], [0])

# compile the circuit down to low-level QASM instructions
# supported by the backend (not needed for simple circuits)
compiled_circuit = transpile(circuit, simulator)

# Execute the circuit on the qasm simulator
job = simulator.run(compiled_circuit, shots=1000000)

# Grab results from the job
result = job.result()

# Returns counts
counts = result.get_counts(circuit)
print("\nTotal count for 0 and 1 are:", counts)

# Draw the circuit
d = circuit.draw(output="mpl")

# Plot outcome histogram
plot_histogram(counts)
plt.show()