# https://github.com/Sammyalhashe/Thesis/blob/c22cff964f1c635eb28be1130c02fe2d95e536c8/Grover/Test/testSqrtGate.py
"""
This file contains test code.
"""
###############################################################################
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
# from qiskit import execute, compile  # available_backends
# from qiskit.tools.visualization import plot_histogram  # plot_state,
from qiskit import Aer  # IBMQ
# from qiskit.backends.jobstatus import JOB_FINAL_STATES
# import Qconfig
from qiskit.tools.visualization import circuit_drawer
from qiskit import execute  # compile  # available_backends
# from matplotlib import pyplot as plt
# import numpy as np
# from scipy import linalg as la
from numpy import array
# from sqrtGate import sqrtGate
from makeGates import nCGATE

###############################################################################
# arrays
x = array([[0, 1], [1, 0]], dtype=complex)

# MS = array(
# [[1, 0, 0, 0 - 1j], [0, 1, 0 - 1j, 0], [0, 0 - 1j, 1, 0],
# [0 - 1j, 0, 0, 1]],
# dtype=complex)

###############################################################################
q = QuantumRegister(5, 'q')
c = ClassicalRegister(5, 'c')

qc = QuantumCircuit(q, c)

# test nControl gate: gate needs to be a 2D matrix
nCGATE(qc, [q[i] for i in range(5)], x)

qc.barrier()

qc.measure(q, c)

circuit_drawer(qc, filename='sqrtGate_test.png')
###############################################################################
backend_sim = Aer.get_backend('qasm_simulator')
job_sim = execute(qc, backend_sim)
result_sim = job_sim.result()
# Show the results
print("simulation: ", result_sim)
counts = result_sim.get_counts(qc)
print(counts)
###############################################################################
