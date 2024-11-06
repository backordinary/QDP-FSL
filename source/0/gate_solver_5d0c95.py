# https://github.com/Zer0112/quantum_job_shop/blob/7ba1b1d943064d5c7a94561001397f558da9f227/src/gate_solver.py
import numpy as np

# importing Qiskit
from qiskit import IBMQ, Aer
from qiskit.providers.ibmq import least_busy
from qiskit import QuantumCircuit, assemble, transpile
# import basic plot tools
from qiskit.visualization import plot_histogram

from qiskit.providers.aer import QasmSimulator

QAOA = QuantumCircuit(1, 1)
simulator = QasmSimulator()

backend = Aer.get_backend("qasm_simulator")
shots = 10000

TQAOA = transpile(QAOA, backend)
qobj = assemble(TQAOA, shots=shots)
QAOA_results = backend.run(qobj).result()
