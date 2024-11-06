# https://github.com/ChistovPavel/QubitExperience/blob/532c2a7c99b0d95878178451e6ab85ac1222debc/QubitExperience/QuantumTeleportation.py
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import BellsState

def teleportateQuantumState(targetState):

    qr = QuantumRegister(3)
    cr1 = ClassicalRegister(1)
    cr2 = ClassicalRegister(1)
    cr3 = ClassicalRegister(1)
    teleportationCircuit = QuantumCircuit(qr, cr1, cr2, cr3)

    teleportationCircuit.initialize(targetState, 0)

    BellsState.getBellsState2(teleportationCircuit, 1, 2)

    teleportationCircuit.cx(0, 1)
    teleportationCircuit.h(0)

    teleportationCircuit.measure(0,0)
    teleportationCircuit.measure(1,1)

    teleportationCircuit.z(2).c_if(cr1, 1)
    teleportationCircuit.x(2).c_if(cr2, 1)

    teleportationCircuit.measure(2,2)

    return teleportationCircuit
