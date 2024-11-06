# https://github.com/ccicconetti/qiskit-tests/blob/8e362a1d041b071b64b9630201e0bb56246c735a/tutorial/tutorial2.py
import numpy as np
import matplotlib.pyplot as plt
from qiskit import(
  QuantumCircuit,
  execute,
  Aer)
from qiskit.visualization import plot_histogram
from qiskit.tools.visualization import (
  plot_state_city,
  plot_bloch_multivector)

# Use Aer's qasm_simulator
#simulator = Aer.get_backend('qasm_simulator')
simulator = Aer.get_backend('statevector_simulator')

# Create a Quantum Circuit acting on the q register
circuit = QuantumCircuit(2)

# Add a H gate on qubit 0
circuit.h(0)

# Add a CX (CNOT) gate on control qubit 0 and target qubit 1
circuit.cx(0, 1)

circuit.t(0)

#circuit.measure_all()

# Execute the circuit on the statevector simulator
job = execute(circuit, simulator)

# Grab results from the job
result = job.result()

statevector = result.get_statevector(circuit)
plot_bloch_multivector(statevector, title='Bloch sphere')
plot_state_city(statevector, title='State vector')
print(statevector)

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