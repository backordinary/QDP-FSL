# https://github.com/lucas-oliveira/quantum-parallelism/blob/d4846cff1f8860ae898442ed96a91671f93489cc/deutsch.py
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.providers.aer import QasmSimulator
from qiskit.visualization import plot_histogram
