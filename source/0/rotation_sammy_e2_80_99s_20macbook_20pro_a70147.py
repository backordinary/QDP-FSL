# https://github.com/Sammyalhashe/Thesis/blob/c22cff964f1c635eb28be1130c02fe2d95e536c8/Grover/Test/Rotation-Sammy%E2%80%99s%20MacBook%20Pro.py
"""
This file contains test code.
"""
###############################################################################
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.tools.visualization import plot_histogram  # plot_state,
# from qiskit import execute, compile  # available_backends
# from qiskit import Aer, IBMQ
# from qiskit.backends.jobstatus import JOB_FINAL_STATES
# import Qconfig
# from qiskit.tools.visualization import circuit_drawer
from qiskit.mapper import euler_angles_1q, two_qubit_kak
from qiskit import Aer  # , IBMQ
from qiskit import execute  # compile  # available_backends
# from matplotlib import pyplot as plt
# import numpy as np
# from scipy import linalg as la
from numpy import pi, array
from sqrtGate import sqrtGate
from math import log, floor

###############################################################################
# Testing values
sqrtX = array(
    [[0.5 * (1 + 1j), 0.5 * (1 - 1j)], [0.5 * (1 - 1j), 0.5 * (1 + 1j)]],
    dtype=complex)
sqrtX2, sqrtX2_I = sqrtGate()
MS = array(
    [[1, 0, 0, 0 - 1j], [0, 1, 0 - 1j, 0], [0, 0 - 1j, 1, 0],
     [0 - 1j, 0, 0, 1]],
    dtype=complex)

theta, phi, lmbda, s = euler_angles_1q(sqrtX)
theta2, phi2, lmbda2, s2 = euler_angles_1q(sqrtX2)
MS_gate = two_qubit_kak(MS)
print(theta, phi, lmbda, s)
print(theta2, phi2, lmbda2, s2)
for gate in MS_gate:
    print(gate)
###############################################################################

# still have to construct the controlled versions of the sqrt gates!


def change_basis(qc, q, theta, phi):
    qc.u3(theta, phi, phi, q)


def R(qc, q, theta, phi):
    """R
    General rotation: Rx * Ry * Rz
    :param qc: QuantumCircuit
    :param q: qubit targeted
    :param theta: theta angle
    :param phi: phi angle
    """
    qc.rx(theta, q)
    qc.ry(theta, q)
    qc.rz(phi, q)


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


def MS_gate2(qc, qubits):
    if len(qubits) != 2:
        raise ValueError("qubits needs to be two qubits")
    q0, q1 = qubits[0], qubits[1]
    qc.x(q1)
    qc.cx(q1, q0)
    qc.s(q1)
    qc.h(q1)
    qc.s(q1)
    qc.cx(q1, q0)


def MS_gaten(qc, qubits):
    if len(qubits) < 2:
        raise ValueError("Need at least to be two qubits")
    last = qubits[-1]
    n = len(qubits)
    for i, q in enumerate(range(n)):
        if i != n - 1:
            qc.cx(last, qubits[q])
    qc.s(last)
    qc.h(last)
    qc.s(last)
    for q in range(n - 2, -1, -1):
        qc.cx(last, qubits[q])


###############################################################################
def nCNOT(qc, qubits):
    """nCNOT

    :param qc: QuantumCircuit to apply this gate to
    :param qubits: qubits to act on. This must be a list, with the last
    element being the target qubit and all other elements being the cotrol
    qubits.
    """
    n = len(qubits)
    if n == 0:
        return
    if n == 1:
        qc.x(qubits)
        return
    if n == 2:
        qc.cx(qubits[0], qubits[1])
        return
    if n == 3:
        qc.ccx(qubits[0], qubits[1], qubits[2])
        return
    nCNOT(qc, qubits[:-1])


###############################################################################
def Rotation_oracle(qc, q, mapping):
    # mapping = {'00': (pi, pi), '01': (pi, 0), '10': (0, pi), '11': (0, 0)}
    # mapping = {'00': (pi, pi), '01': (0, pi), '10': (pi, 0), '11': (0, 0)}
    # Oracle
    tup = mapping[str(index)]
    # alpha, beta = mapping[str(index)]
    for ind, a in enumerate(q):
        R(qc, a, tup[ind], 0)

    for a in q:
        R(qc, a, pi / 2, pi)

    # R(qc, q[0], alpha, 0)
    # R(qc, q[1], beta, 0)
    # R(qc, q[0], pi / 2, pi)
    # R(qc, q[1], pi / 2, pi)

    qc.barrier()

    # MS Gate
    MS_gaten(qc, [a for a in q])

    qc.barrier()

    for a in q:
        R(qc, a, pi / 2, 0)

    # R(qc, q[0], pi / 2, 0)
    # R(qc, q[1], pi / 2, 0)

    qc.barrier()

    for a in q:
        qc.rz(-pi / 2, a)

    # qc.rz(-pi / 2, q[0])
    # qc.rz(-pi / 2, q[1])

    qc.barrier()

    for ind, a in enumerate(q):
        R(qc, a, tup[ind], 0)

    # R(qc, q[0], alpha, 0)
    # R(qc, q[1], beta, 0)


###############################################################################


def Rotation_diffusion(qc, q):
    for a in q:
        R(qc, a, pi / 2, -pi / 2)

    # R(qc, q[0], pi / 2, -pi / 2)
    # R(qc, q[1], pi / 2, -pi / 2)

    qc.barrier()

    MS_gaten(qc, [a for a in q])


###############################################################################
# IBMQ.load_accounts()
# print(IBMQ.backends())

###############################################################################
index = '101'
n = len(index)
# max number the number of digits can represent
max_num = 2**n - 1
num_digits_repre = floor(log(max_num) / log(2) + 1)
mapping = {}
for i in range(max_num + 1):
    binary = "{0:b}".format(i)
    if len(binary) < num_digits_repre:
        adding = "0" * (num_digits_repre - len(binary))
        binary = adding + binary
    tup = []
    for i in range(len(binary)):
        if binary[i] == '0':
            tup.append(pi)
        else:
            tup.append(0)
    mapping[binary] = tup
print(mapping)
# mapping = {'00': (pi, pi), '01': (0, pi), '10': (pi, 0), '11': (0, 0)}

q = QuantumRegister(n, 'q')
c = ClassicalRegister(n, 'c')

qc = QuantumCircuit(q, c)

# "Hadamard" implementation
for i in range(n):
    R(qc, q[i], pi / 2, 0)
# R(qc, q[0], pi / 2, 0)
# R(qc, q[1], pi / 2, 0)
qc.barrier()

# alpha, beta = mapping[str(index)]
# change_basis(qc, q[0], alpha, 0)
# change_basis(qc, q[1], beta, 0)
for i in range(len(index)):
    Rotation_oracle(qc, q, mapping)

    qc.barrier()
    Rotation_diffusion(qc, q)
    qc.barrier()

# nCNOT(qc, [q[i] for i in range(2)])
# nCNOT(qc, [q[i] for i in range(1, 3)])

qc.measure(q, c)

# circuit_drawer(qc, filename='rotation_test.png')
###############################################################################
backend_sim = Aer.get_backend('qasm_simulator')
job_sim = execute(qc, backend_sim)
result_sim = job_sim.result()
# Show the results
print("simulation: ", result_sim)
counts = result_sim.get_counts(qc)
print(counts)
plot_histogram(counts)
###############################################################################
