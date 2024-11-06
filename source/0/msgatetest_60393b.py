# https://github.com/Sammyalhashe/Thesis/blob/c22cff964f1c635eb28be1130c02fe2d95e536c8/Grover/Test/MSGateTest.py
"""
This file contains test code to build the MS Gate.
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
from numpy import pi, array
from makeGates import makeGatesFrom4DMatrix

###############################################################################
MS = array(
    [[1, 0, 0, 0 - 1j], [0, 1, 0 - 1j, 0], [0, 0 - 1j, 1, 0],
     [0 - 1j, 0, 0, 1]],
    dtype=complex)


###############################################################################
def MS_gate(qc, qubits):
    if len(qubits) != 2:
        raise ValueError("qubits needs two qubits")
    q0, q1 = qubits[0], qubits[1]
    qc.iden(q0)
    qc.u1(-pi / 2, q1)
    qc.cx(q1, q0)
    qc.u1(pi / 2, q0)
    qc.u3(-pi, 0, 0, q1)
    qc.cx(q0, q1)
    qc.u2(0, 0, q1)
    qc.cx(q1, q0)
    qc.u1(pi / 2, q0)
    qc.iden(q1)


###############################################################################
q = QuantumRegister(2, 'q')
c = ClassicalRegister(2, 'c')

qc = QuantumCircuit(q, c)

MS_gate(qc, [q[0], q[1]])

qc.barrier()

makeGatesFrom4DMatrix(qc, [q[0], q[1]], MS)

qc.barrier()

qc.measure(q, c)

qc.barrier()
circuit_drawer(qc, filename='MSGate_test.png')
