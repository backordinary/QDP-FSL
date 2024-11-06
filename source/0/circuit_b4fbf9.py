# https://github.com/myknlsn/qosf-task1/blob/61d589fa1b0b2a8271ca73a8272119f8957c69ff/circuit.py
#initialization
import matplotlib.pyplot as plt
import numpy as np

# importing Qiskit
from qiskit import IBMQ, Aer, assemble, transpile
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.providers.ibmq import least_busy

# import basic plot tools
from qiskit.visualization import plot_histogram

#n = 30
#task_circuit = QuantumCircuit(n)  #Circuit size for the task. I don't know how to optimize this for now. 


#Initializes circuit: address qubits, fan reader and minus ancilla. 
def initialize(qc):  
    for q in [0,1]:
        qc.h(q)
    qc.x(2)
    qc.x(29)
    qc.h(29)
    return qc

#Load data onto registers in sequence
def data_load(qc, z): # circuit, List of integers
    bit_to_qubit = ""
    for a in z: 
        t = bin(a)[2:]
        while len(t) < 4:
            t = "0"+t
        bit_to_qubit = bit_to_qubit+t
    for i in range(16):
        if bit_to_qubit[i] == '1':
            qc.x(i+6)   # Offset of 6 shifts to first qubit in register
    return qc

#Create a superposition of data entangled to the addresses
def data_super(qc):
    qc.cx(1,3)
    qc.cx(3,2)
    qc.ccx(0,2,4)
    qc.ccx(0,3,5)
    qc.cx(4,2)
    qc.cx(5,3)
    qc.ccx(0 + 2, (0*4)+ 0 + 6, 0 + 22)
    qc.ccx(0 + 2, (0*4)+ 1 + 6, 1 + 22)
    qc.ccx(0 + 2, (0*4)+ 2 + 6, 2 + 22)
    qc.ccx(0 + 2, (0*4)+ 3 + 6, 3 + 22)
    qc.ccx(1 + 2, (1*4)+ 0 + 6, 0 + 22)
    qc.ccx(1 + 2, (1*4)+ 1 + 6, 1 + 22)
    qc.ccx(1 + 2, (1*4)+ 2 + 6, 2 + 22)
    qc.ccx(1 + 2, (1*4)+ 3 + 6, 3 + 22)
    qc.ccx(2 + 2, (2*4)+ 0 + 6, 0 + 22)
    qc.ccx(2 + 2, (2*4)+ 1 + 6, 1 + 22)
    qc.ccx(2 + 2, (2*4)+ 2 + 6, 2 + 22)
    qc.ccx(2 + 2, (2*4)+ 3 + 6, 3 + 22)
    qc.ccx(3 + 2, (3*4)+ 0 + 6, 0 + 22)
    qc.ccx(3 + 2, (3*4)+ 1 + 6, 1 + 22)
    qc.ccx(3 + 2, (3*4)+ 2 + 6, 2 + 22)
    qc.ccx(3 + 2, (3*4)+ 3 + 6, 3 + 22)
    return qc

def data_super_reverse(qc):
    qc.ccx(5 - 0, 21 - (0*4+ 0), 22- 0)
    qc.ccx(5 - 0, 21 - (0*4+ 1), 22- 1)
    qc.ccx(5 - 0, 21 - (0*4+ 2), 22- 2)
    qc.ccx(5 - 0, 21 - (0*4+ 3), 22- 3)
    qc.ccx(5 - 1, 21 - (1*4+ 0), 22- 0)
    qc.ccx(5 - 1, 21 - (1*4+ 1), 22- 1)
    qc.ccx(5 - 1, 21 - (1*4+ 2), 22- 2)
    qc.ccx(5 - 1, 21 - (1*4+ 3), 22- 3)
    qc.ccx(5 - 2, 21 - (2*4+ 0), 22- 0)
    qc.ccx(5 - 2, 21 - (2*4+ 1), 22- 1)
    qc.ccx(5 - 2, 21 - (2*4+ 2), 22- 2)
    qc.ccx(5 - 2, 21 - (2*4+ 3), 22- 3)
    qc.ccx(5 - 3, 21 - (3*4+ 0), 22- 0)
    qc.ccx(5 - 3, 21 - (3*4+ 1), 22- 1)
    qc.ccx(5 - 3, 21 - (3*4+ 2), 22- 2)
    qc.ccx(5 - 3, 21 - (3*4+ 3), 22- 3)
    qc.cx(5,3)
    qc.cx(4,2)
    qc.ccx(0,3,5)
    qc.ccx(0,2,4)
    qc.cx(3,2)
    qc.cx(1,3)
    return qc

#Call an oracle that kicks back a phase for good addresses
def oracle(qc):
    qc.cx(22,23)
    qc.cx(23,26)
    qc.cx(26,27)
    qc.cx(23,26)
    qc.cx(22,23)
    qc.cx(23,24)
    qc.ccx(24,27,28)
    qc.cx(24,27)
    qc.cx(23,24)
    qc.cx(24,25)
    qc.ccx(25,28,29)
    qc.cx(25,28)
    qc.cx(24,25)

# Borrowed diffuser here
def diffuser(qc):
    qc.h([0,1])
    qc.z([0,1])
    qc.cz(0,1)
    qc.h([0,1])
    return qc

#Note to self: circuit object methods apply gate operations in the sequence they are called. 
#Once oracle design is finalized, reorder code to desire structure, consider modifications for extension. 