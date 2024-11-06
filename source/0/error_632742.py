# https://github.com/gwjacobson/QuantumErrorCorrection/blob/6d8cb6c3ed10399575c562ce76b8131c6f799858/error.py
from qiskit import *
from qiskit.quantum_info.operators import Operator
import random
import numpy as np
from numpy import pi

## Functions to define different error models inspired by Chandak, Mardia, and Tolunay

#Include a Bit Flip, Phase Flip, Y Rotation, Hadamard, and Arbitrary Rotation

#function to add bit_flip error
def bit_flip(circuit, qubit):
    circuit.x(qubit)
    return

#function to add phase_flip error
def phase_flip(circuit, qubit):
    circuit.h(qubit)
    circuit.z(qubit)
    return

#function to add phase and bit error
def y_error(circuit, qubit):
    circuit.y(qubit)

#function for arbitrary single qubit error
#will apply 3 of the 5 gates for random error
def arbitrary_error(circuit, qubit):
    e = random.randint(1,5)
    if e == 1:
        circuit.id(qubit) #identity
    elif e == 2:
        circuit.x(qubit) #bit flip
    elif e == 3:
        circuit.y(qubit) #bit and phase flip
    else:
        circuit.z(qubit) #phase flip
    e = random.randint(1,5)
    if e == 1:
        circuit.id(qubit) #identity
    elif e == 2:
        circuit.x(qubit) #bit flip
    elif e == 3:
        circuit.y(qubit) #bit and phase flip
    else:
        circuit.z(qubit) #phase flip
    e = random.randint(1,5)
    if e == 1:
        circuit.id(qubit) #identity
    elif e == 2:
        circuit.x(qubit) #bit flip
    elif e == 3:
        circuit.y(qubit) #bit and phase flip
    else:
        circuit.z(qubit) #phase flip

    return

