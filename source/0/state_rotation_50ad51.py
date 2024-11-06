# https://github.com/lucassciuti/qiskit/blob/d91e1ccfe069771d780ea3f205fc827563406f86/qiskit_intro/state_rotation.py
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit.providers.aer import QasmSimulator
from qiskit.visualization import plot_histogram, plot_bloch_vector

simulator = QasmSimulator()

circuit = QuantumCircuit(1)
circuit.rz(0)

compiled_circuit = transpile(circuit, simulator)

print(circuit.draw())