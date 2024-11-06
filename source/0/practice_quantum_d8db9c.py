# https://github.com/ashpmath/Quantum-Computing/blob/ef21a64df6816d2750525beeeb6ea42c0ef9dfc7/Practice/Python/Practice_Quantum.py
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 17:38:37 2021

@author: Ashley P. Mathews
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.providers.aer import QasmSimulator
from qiskit.visualization import plot_histogram
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.tools.visualization import circuit_drawer
from qiskit.visualization import plot_histogram

# Use Aer's qasm_simulator
simulator = QasmSimulator()

# Create a Quantum Circuit acting on the q register
circuit = QuantumCircuit(2, 2)

# Add a H gate on qubit 0
circuit.h(0)

# Add a CX (CNOT) gate on control qubit 0 and target qubit 1
circuit.cx(0, 1)
circuit.cx(1,0)

# Map the quantum measurement to the classical bits
circuit.measure([0,1], [0,1])

# compile the circuit down to low-level QASM instructions
# supported by the backend (not needed for simple circuits)
compiled_circuit = transpile(circuit, simulator)

# Execute the circuit on the qasm simulator
job = simulator.run(compiled_circuit, shots=1000)

# Grab results from the job
result = job.result()

# Returns counts
counts = result.get_counts(compiled_circuit)
print("\nTotal count for 00 and 11 are:",counts)

# q = QuantumRegister(1)
# c = ClassicalRegister(1)
# qc = QuantumCircuit(q, c)
# qc.h(q)
# qc.measure(q, c)
# qc.draw(output='mpl', style={'backgroundcolor': '#EEEEEE'})

# Draw the circuit
circuit.draw(circuit)
plot_histogram(counts)

# Plot the histogram 
plot_histogram(circuit)


