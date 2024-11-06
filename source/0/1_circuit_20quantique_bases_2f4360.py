# https://github.com/SofianeChalal/Quantum/blob/bc5b3f682cb020c6b251321366802fc2e2a217c3/1-Circuit%20Quantique[Bases].py
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 13:27:45 2021

@author: Sofiane_C
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.providers.aer import QasmSimulator
from qiskit.visualization import plot_histogram

# Aer's qasm_simulator
simulator = QasmSimulator()

# Création d'un circuit Quantique
circuit = QuantumCircuit(2, 2)

# Activation de la porte d'Hadamard au qubit 0
circuit.h(0)

# Porte CNOT (CX) qubit  de controle 0 et qubit 1 comme cible
circuit.cx(0, 1)

# Measure des résultat
circuit.measure([0,1], [0,1])


compiled_circuit = transpile(circuit, simulator)

# Execution du circuit.
job = simulator.run(compiled_circuit, shots=1000)

# Saisir les résultats du travail
result = job.result()

# Comptage
counts = result.get_counts(compiled_circuit)
print("\nCompte totale pour 00 et 11  : ",counts)

# Draw the circuit
circuit.draw()

# Tracé de l'histograme
plot_histogram(counts)
