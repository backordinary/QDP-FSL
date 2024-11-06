# https://github.com/Sammyalhashe/Thesis/blob/c22cff964f1c635eb28be1130c02fe2d95e536c8/Grover/Test/cCGATE_test.py
"""
This file contains test code to test the cCGATE function in
the makeGates.py file.
"""
###############################################################################
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
# from qiskit import execute, compile  # available_backends
# from qiskit.tools.visualization import plot_histogram  # plot_state,
# from qiskit import Aer, IBMQ
# from qiskit.backends.jobstatus import JOB_FINAL_STATES
# import Qconfig
from qiskit.tools.visualization import circuit_drawer
# from matplotlib import pyplot as plt
# import numpy as np
# from scipy import linalg as la
from numpy import array
from makeGates import nCGATE

###############################################################################
X = array([[0, 1], [1, 0]])
q = QuantumRegister(2, 'q')
c = ClassicalRegister(2, 'c')

qc = QuantumCircuit(q, c)

nCGATE(qc, [q[i] for i in range(2)], X)

qc.measure(q, c)

qc.barrier()

circuit_drawer(qc, filename='nCGATE_test.png')
