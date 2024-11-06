# https://github.com/teaguetomesh/VQE/blob/7c3afde179e4bc304e4163471fe521c1121c7d37/Ansatz/naive_ansatz.py
'''
Teague Tomesh - 3/13/2019

Implementation of a naive ansatz for use in the VQE algorithm.
Adapted from [EPiQC VQE tutorial by Pranav Gokhale]
(https://www.youtube.com/watch?v=E947xs9-Mso)

In the tutorial, this ansatz was designed for 2 qubits, here I am extending
it to 4 qubits. 

'''

from qiskit import QuantumCircuit, QuantumRegister
import sys


def genCircuit(M, p):
    '''
    '''
    if M is not 4:
        print('ERROR: The naive ansatz is currently implemented for 4 qubits', 
              ' only')
        sys.exit()

    # Initialize quantum register and circuit
    qr = QuantumRegister(M, name='qreg')
    c  = QuantumCircuit(qr, name='naive_ansatz')
    c.rx(p[0], qr[0])
    c.rz(p[0+4], qr[0])
    c.rx(p[1], qr[1])
    c.rz(p[1+4], qr[1])
    c.rx(p[2], qr[2])
    c.rz(p[2+4], qr[2])
    c.rx(p[3], qr[3])
    c.rz(p[3+4], qr[3])
    # ladder down
    c.h(qr[0])
    c.cx(qr[0],qr[0+1])
    # ladder down
    c.h(qr[1])
    c.cx(qr[1],qr[1+1])
    # ladder down
    c.h(qr[2])
    c.cx(qr[2],qr[2+1])
    c.h(qr[3])
    for i in range(3,0,-1):
        # ladder up
        c.cx(qr[i],qr[i-1])

    c.barrier(qr)
    c.rx(p[0+8] ,qr[0])
    c.rz(p[0+12],qr[0])
    c.rz(p[0+16],qr[0])
    c.rx(p[1+8] ,qr[1])
    c.rz(p[1+12],qr[1])
    c.rz(p[1+16],qr[1])
    c.rx(p[2+8] ,qr[2])
    c.rz(p[2+12],qr[2])
    c.rz(p[2+16],qr[2])
    c.rx(p[3+8] ,qr[3])
    c.rz(p[3+12],qr[3])
    c.rz(p[3+16],qr[3])

    return c



