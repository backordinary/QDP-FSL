# https://github.com/kgeorgopoulos2/programbench/blob/d083b3e7f500dc599b53effdf02c5cc9cf9070fd/qsearch.py
# Import the necessary libraries
import numpy as np
from math import pi as pi

# importing QISKit
from qiskit import Aer, IBMQ, BasicAer
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, execute

def cccz(qc, q):
    '''Method that makes a cccz with no ancilla qubits.'''
    qc.cp(pi/4, q[0], q[3])
    qc.cx(q[0], q[1])
    qc.cp(-pi/4, q[1], q[3])
    qc.cx(q[0], q[1])
    qc.cp(pi/4, q[1], q[3])
    qc.cx(q[1], q[2])
    qc.cp(-pi/4, q[2], q[3])
    qc.cx(q[0], q[2])
    qc.cp(pi/4, q[2], q[3])
    qc.cx(q[1], q[2])
    qc.cp(-pi/4, q[2], q[3])
    qc.cx(q[0], q[2])
    qc.cp(pi/4, q[2], q[3])

    return qc

# Toffoli gate for n qubits
def ncz(circ, tgt, ctrl, anc):
    '''Create a generalized controlled Z gate with the use of ancilla qubits.'''
    # Compute
    circ.ccx(ctrl[0], ctrl[1], anc[0])
    for i in range(2, len(ctrl)):
        circ.ccx(ctrl[i], anc[i-2], anc[i-1])
    
    # CNOT
    circ.cz(anc[len(ctrl)-2], tgt)

    # Uncompute
    for i in range(len(ctrl)-1, 1, -1):
        circ.ccx(ctrl[i], anc[i-2], anc[i-1])
    circ.ccx(ctrl[0], ctrl[1], anc[0])
    
    return circ

def groverAnc(q, anc, qc, n, iterations):
    '''Creates the Grover iteration of the algorithm.'''
    tgt = q[n-1]
    ctl = list(q[0:n-1])
    
    for i in range(0,iterations):
        # Oracle for the marked state |1010>
        qc.x(q[0])
        qc.x(q[2])
        qc = ncz(qc, tgt, ctl, anc)
        qc.x(q[0])
        qc.x(q[2])

        # Amplification
        for i in range(0, n):
            qc.h(q[i])

        for i in range(0, n):
            qc.x(q[i])

        qc = ncz(qc, tgt, ctl, anc)

        for i in range(0, n):
            qc.x(q[i])

        for i in range(0, n):
            qc.h(q[i])
            
        qc.barrier()
    
    return qc

def grover(q, qc, n, iterations):
    '''Creates the Grover iteration of the algorithm.'''
    tgt = q[n-1]
    ctl = list(q[0:n-1])
    
    for i in range(0,iterations):
        # Oracle for the marked state |1010>
        qc.x(q[0])
        qc.x(q[2])
        qc = cccz(qc, q)
        qc.x(q[0])
        qc.x(q[2])

        # Amplification
        for i in range(0, n):
            qc.h(q[i])

        for i in range(0, n):
            qc.x(q[i])

        qc = cccz(qc, q)

        for i in range(0, n):
            qc.x(q[i])

        for i in range(0, n):
            qc.h(q[i])
            
        qc.barrier()
    
    return qc

def groverSearch(n, iterations, ancilla):
    '''Returns the quantum circuit for a grover search on 4 qubits for the item |1010>'''
    if (ancilla == True):
        q = QuantumRegister(n, 'q')
        anc = QuantumRegister(n-2, 'anc')
        c = ClassicalRegister(n, 'c')
        qc = QuantumCircuit(q, anc, c)

        # Initialization
        for i in range(0, n):
            qc.h(q[i])

        qc = groverAnc(q, anc, qc, n, iterations)
    else:
        q = QuantumRegister(n, 'q')
        c = ClassicalRegister(n, 'c')
        qc = QuantumCircuit(q, c)

        # Initialization
        for i in range(0, n):
            qc.h(q[i])

        qc = grover(q, qc, n, iterations)

    # Measure
    qc.measure(q, c)
    
    return qc