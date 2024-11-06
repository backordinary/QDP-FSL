# https://github.com/Crabster/qiskit-learning/blob/3f14c39ee294f42e3f83a588910b659280556a68/circuits/common_gates.py
import qiskit

from math import pi
from random import random

def random_state_gate():
    theta = 2*pi*random()
    phi = 2*pi*random()
    lam = 2*pi*random()

    qc = qiskit.QuantumCircuit(1)
    qc.u3(theta, phi, lam, 0)
    gate = qc.to_gate()
    gate.name = f"$U_3$ {theta:.2f},{phi:.2f},{lam:.2f}"
    return gate

def phi_plus_gate():
    qc = qiskit.QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    gate = qc.to_gate()
    gate.name = "$\phi^{+}$"
    return gate

def phi_minus_gate():
    qc = qiskit.QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.x(0)
    qc.z(0)
    gate = qc.to_gate()
    gate.name = "$\phi^{-}$"
    return gate

def psi_plus_gate():
    qc = qiskit.QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.x(0)
    gate = qc.to_gate()
    gate.name = "$\psi^{+}$"
    return gate

def psi_minus_gate():
    qc = qiskit.QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.z(0)
    gate = qc.to_gate()
    gate.name = "$\psi^{-}$"
    return qc
