# https://github.com/ChistovPavel/QubitExperience/blob/532c2a7c99b0d95878178451e6ab85ac1222debc/QubitExperience/BellsState.py
from qiskit import QuantumCircuit, assemble, Aer

def getBellsState1(initialVector1, initialVector2):
    qc = QuantumCircuit(2)
    qc.initialize(initialVector1, 0)
    qc.initialize(initialVector2, 1)
    qc.h(0)
    qc.cx(0, 1)
    return qc

def getBellsState2(qc, firstQubitIndex, secondQubitIndex):
    qc.h(firstQubitIndex)
    qc.cx(firstQubitIndex, secondQubitIndex)
    return qc