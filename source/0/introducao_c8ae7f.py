# https://github.com/rodrigopff/ibm_quantica/blob/2261e534a15864e1abef397129d26c9e65c70a30/introducao.py
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.providers.aer import QasmSimulator
from qiskit.visualization import plot_histogram
from qiskit.visualization import circuit_drawer 
from qiskit.visualization import plot_bloch_vector
import matplotlib.pyplot as plt


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
job = simulator.run(compiled_circuit, shots=100000)

# Grab results from the job
result = job.result()

# Returns counts
counts = result.get_counts(compiled_circuit)
print("\nTotal count for 00 and 11 are:",counts)

# Draw the circuit
# circuit_drawer(cir'cuit, output='mpl', style={'backgroundcolor': '#EEEEEE'})
circuit.draw('mpl')

# Plot a histogram
plot_histogram(counts)
# plot_bloch_vector([0,0,1], title="")
# plot_bloch_vector([np.sqrt(2),np.pi/4 ,np.pi/4], title="")

plt.show()