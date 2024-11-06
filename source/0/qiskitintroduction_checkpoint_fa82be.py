# https://github.com/lancecarter03/QiskitLearning/blob/a0fd28b0a55645177b8df3cee447849e8c16d9cf/.ipynb_checkpoints/qiskitIntroduction-checkpoint.py
import numpy as np
from qiskit import QuantumCircuit
from qiskit.providers.aer import QasmSimulator
from qiskit.visualization import plot_histogram

circuit = QuantumCircuit(2,2)

circuit.h(0)
circuit.cx(0,1)
circuit.measure([0,1],[0,1])
circuit.draw()
