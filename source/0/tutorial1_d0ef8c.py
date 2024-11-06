# https://github.com/ccicconetti/qiskit-tests/blob/8e362a1d041b071b64b9630201e0bb56246c735a/tutorial/tutorial1.py
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
circuit = QuantumCircuit(2, 2)

# Add a H gate on qubit 0
circuit.h(0)

# Add a CX (CNOT) gate on control qubit 0 and target qubit 1
circuit.cx(0, 1)

circuit.h(0)

circuit.t(0)

circuit.h(0)

#
# the resulting entangled state is
#
#              /                \
#             | 1 + exp(i*pi/4) |
#             | 1 - exp(i*pi/4) |
# |psi> = 1/4 |                 |
#             | 1 - exp(i*pi/4) | 
#             | 1 + exp(i*pi/4) |
#              \                /
#
#
# P(00, 11) =~ 0.427
# P(01, 10) =~ 0.074
#

# Map the quantum measurement to the classical bits
circuit.measure([0,1], [0,1])

# Execute the circuit on the qasm simulator
job = execute(circuit, simulator, shots=10000)

# Grab results from the job
result = job.result()

# Returns counts
counts = result.get_counts(circuit)
print("\nCounts:",counts)

# Draw the circuit
circ_plt = circuit.draw(output='mpl')
plt.show(block=False)
print(circuit.draw(output='text'))

# Plot a histogram
fig = plot_histogram(counts)

plt.show(block=True)