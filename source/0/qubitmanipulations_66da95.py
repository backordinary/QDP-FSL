# https://github.com/aapoorv-tf/qiskit-quantumcomputing/blob/f605e22db891204045201e69158738b2f289cc6c/qubitmanipulations.py
from qiskit import *
from math import pi
from qiskit.visualization import plot_bloch_multivector

#X-gate
qc = QuantumCircuit(1)
qc.x(0)
print(qc.draw('text'))

# backend  = Aer.get_backend('statevector_simulator')
# out = execute(qc, backend).result().get_statevector()
# print(plot_bloch_multivector(out))

#hadamard
qc.h(0)
print(qc.draw('text'))