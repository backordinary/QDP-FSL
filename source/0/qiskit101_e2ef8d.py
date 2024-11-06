# https://github.com/kunal077/Quantum.Computation/blob/f468ad462ef929227ae4d0c4474f6cad1b1aed9b/qiskit101.py
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.providers.aer import QasmSimulator
from qiskit.visualization import plot_histogram

simulator = QasmSimulator()
circuit - QuantumCircuit(2, 2)
circuit.h(0)

circuit.cx(0, 1)
circuit.measure([0, 1], [0, 1])

compiled_circuit = transpile(circuit, simulator)
job = simulator.run(compiled_circuit, shots=1000)
result = job.result()

counts = result.get_counts(circuit)
print("\nTotal count for 00 and 11 are:",counts)
circuit.draw()
