# https://github.com/xXxSpicyBoiiixXx/Quipper-Language/blob/c4f04e6e42406ee251d74148fe9a699e40934b93/Qiskit/workflow-example.py
import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, transpile
from qiskit.providers.aer import QasmSimulator
from qiskit.visualization import plot_histogram


# Aer's qasm_simulator
simulator = QasmSimulator()

# Creating a Quantum Circuit acting on the q register
circuit = QuantumCircuit(2, 2)

# H gate on quibit 0
circuit.h(0)

# CNOT gat on control qubit 0 and target qubit 1
circuit.cx(0, 1)

# Mapping Quantum measurements to classical bits
circuit.measure([0,1], [0,1])

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots = 1000)

result = job.result()

counts = result.get_counts(compiled_circuit)
print("\nTotal count for 00 and 11 are:", counts)

circuit.draw()

plot_histogram(counts)

plt.show()
