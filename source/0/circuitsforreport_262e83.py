# https://github.com/Sammyalhashe/Thesis/blob/c22cff964f1c635eb28be1130c02fe2d95e536c8/Grover/Test/circuitsForReport.py
###############################################################################
from qiskit import QuantumCircuit, QuantumRegister  # ClassicalRegister
# from qiskit import execute, compile  # available_backends
# from qiskit.tools.visualization import plot_histogram  # plot_state,
# from qiskit import Aer, IBMQ
# from qiskit.backends.jobstatus import JOB_FINAL_STATES
# import Qconfig
from qiskit.tools.visualization import circuit_drawer
# from qiskit.mapper import euler_angles_1q, two_qubit_kak
# from qiskit import Aer, IBMQ
# from qiskit import execute  # compile  # available_backends
# from matplotlib import pyplot as plt
# import numpy as np
# from scipy import linalg as la
# from numpy import pi, array
# from sqrtGate import sqrtGate

###############################################################################
q = QuantumRegister(3, 'q')
# c = ClassicalRegister(2, 'c')

qc = QuantumCircuit(q)

qc.h(q)
qc.barrier()
qc.x(q[0])
qc.barrier()
qc.z(q[1])
qc.barrier()
qc.cx(q[0], q[1])
circuit_drawer(qc, filename='basic_circuit.png')
###############################################################################

q1 = QuantumRegister(5, 'q')

qc1 = QuantumCircuit(q1)


