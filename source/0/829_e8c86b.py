# https://github.com/LoganLieou/Leetcode/blob/b3264017b352aa92c940cea90763fbb8d36ed8f2/scratch/829.py
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.providers.aer import QasmSimulator
from qiskit.visualization import plot_histogram

simulator = QasmSimulator()
circuit = QuantumCircuit(2, 2)
circuit.h(0)
circuit.cx(0, 1)
circuit.measure([0, 1], [0, 1])
compiled_circut = transpile(circuit, simulator)

# so concerned
job = simulator.run(compiled_circut, shots=1000)
result = job.result()
counts = result.get_counts(compiled_circut)
print("\nTotal count for 00 and 11 are:", counts)

