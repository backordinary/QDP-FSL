# https://github.com/Sammyalhashe/Thesis/blob/c22cff964f1c635eb28be1130c02fe2d95e536c8/Grover/Test/flip_phase.py
import sys
if sys.version_info < (3, 5):
    raise Exception('Run with python 3')

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import available_backends, execute
from qiskit.tools.visualization import plot_histogram, plot_state
from qiskit import IBMQ, Aer
import Qconfig
from qiskit.tools.visualization import circuit_drawer
from matplotlib import pyplot as plt
import numpy as np
from scipy import linalg as la
from math import sqrt

n = 4
q = QuantumRegister(n, 'q')
c = ClassicalRegister(n, 'c')
anc = QuantumRegister(n-1, 'anc')

qc = QuantumCircuit(q, c, anc)

# setup initial state
qc.x(q[0])
qc.x(q[1])

qc.cx(q[0], anc[0])
qc.ccx(q[1], anc[0], anc[1])
qc.ccx(q[2], anc[1], anc[2])
qc.cx(anc[2], q[3])

qc.measure(q, c)

# compile and run the Quantum circuit on a simulator backend
backend_sim = Aer.get_backend('qasm_simulator')
job_sim = execute(qc, backend_sim)
result_sim = job_sim.result()

# Show the results
print("simulation: ", result_sim)
print(result_sim.get_counts(qc))

circuit_drawer(qc, filename='phase_flip.png')
