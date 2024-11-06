# https://github.com/dschneck/COT5600_schneck/blob/4a7ab7b0479d80fb180eecf6952f46b0d7be56fd/P1/intro.py
#! /usr/bin/env python3

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.providers.aer import QasmSimulator
from qiskit.visualization import plot_histogram

simulator = QasmSimulator()

circuit = QuantumCircuit(2, 2)

circuit.h(0)

circuit.cx(0, 1)

circuit.measure([0,1], [0,1])

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(compiled_circuit)
print("\nTotal count for 00 and 11 are:", counts)

circuit.draw()

plot_histogram(counts)
