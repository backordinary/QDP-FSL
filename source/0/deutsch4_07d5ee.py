# https://github.com/Alvaradom08/reporte-final-/blob/6c013c8a25d408bf001b10c293608070fa56419c/Deutsch4.py
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit import Aer
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
# Use Aer's qasm_simulator
simulator = Aer.get_backend('qasm_simulator')
# Create a Quantum Circuit acting on the q register
circuit = QuantumCircuit(2, 1)

circuit.x(1)
circuit.barrier()

circuit.h(0)
circuit.h(1)
circuit.barrier()


circuit.cx(0, 1)


circuit.barrier()
circuit.h(0)
circuit.barrier()

#Medidas resultado de la funcion
#circuit.measure([0,1], [1,0])
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
print(counts)
# Draw the circuit
print(circuit)
# Plot a histogram
plot_histogram(counts)
plt.show()