# https://github.com/natashaval/qiskit-intro/blob/93f31f6a01c478bd0e730a90ea16973df7c36170/Introduction.py
import numpy as np
from qiskit import QuantumCircuit
from qiskit.providers.aer import QasmSimulator
from qiskit.visualization import plot_histogram

circuit = QuantumCircuit(2,2) # 2 qubits and 2 classical bits
circuit.h(0)
circuit.cx(0,1)
circuit.draw()