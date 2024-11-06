# https://github.com/jcontreras2693/CNYT-Tarea-5/blob/d6a921437714e0bc2f75708d5368002262990aff/Comandos%20%C3%9Atiles.py
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit import Aer
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt


# Use Aer's qasm_simulator
# simulator = Aer.get_backend('qasm_simulator')

# Se leen ambos qubits
# circuit = QuantumCircuit(2, 1)

# Se lee solo el primer qubit
# circuit = QuantumCircuit(2, 1)

# Se lee solo el primer qubit
# circuit = QuantumCircuit(5, 4)

# Map the quantum measurement to the classical bits
# circuit.measure([0], [0])

# Map the quantum measurement to the classical bits
# circuit.measure([0, 1], [1, 0])

# Add a H gate on qubit 0
# circuit.h(0)

# Add a CX (CNOT) gate on control qubit 0 and target qubit 1
# circuit.cx(0, 1)

# Add a barrier on qubit 0 and 1
# circuit.barrier(0, 1)

# Add a barrier on every qubit
# circuit.barrier()

# Add a X (NOT) gate con qubit 0
# circuit.x(0)

# Add a I (Identity) gate on qubit 0
#circuit.i(0)

# compile the circuit down to low-level QASM instructions
# supported by the backend (not needed for simple circuits)
#compiled_circuit = transpile(circuit, simulator)

# Execute the circuit on the qasm simulator
#job = simulator.run(compiled_circuit, shots=1000)

# Grab results from the job
#result = job.result()

# Returns counts
#counts = result.get_counts(circuit)
#print("\nTotal count for 00 and 11 are:", counts)

# Draw the circuit
#print(circuit)

# Plot a histogram
#plot_histogram(counts)
#plt.show()
