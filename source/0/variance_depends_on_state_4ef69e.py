# https://github.com/rsln-s/Importance-of-Kernel-Bandwidth-in-Quantum-Machine-Learning/blob/08e60578f35df1bda4878db64119765e30ed31ef/code/variance_depends_on_state.py
"""
Variance of |<0|psi>|^2 depends on |psi>
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.providers.aer import AerSimulator

nqubits = 10

# first, GHZ
qc = QuantumCircuit(nqubits)
qc.h(0)
for i in range(1,nqubits):
    qc.cx(0,i)
qc.measure_all()
backend = AerSimulator()

overlaps = []

for _ in range(20):
    result = backend.run(qc).result().get_counts()
    overlaps.append(result.get("0"*nqubits, 0) / sum(result.values()))

print("GHZ:", np.std(overlaps))

# second, uniform superposition
qc = QuantumCircuit(nqubits)
for i in range(nqubits):
    qc.h(i)
qc.measure_all()
backend = AerSimulator()

overlaps = []

for _ in range(20):
    result = backend.run(qc).result().get_counts()
    overlaps.append(result.get("0"*nqubits, 0) / sum(result.values()))

print("Uniform:", np.std(overlaps))
