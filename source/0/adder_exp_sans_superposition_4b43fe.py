# https://github.com/danielhutama/ArithmeticQC/blob/87f8481873ea2c8f11ebff83466974a4dbcf673a/ADDER/ADDER_EXP_sans_superposition.py
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 07:57:27 2022


This work is based off of Vlatko Vedral's paper on quantum arithmetic circuits. There is also a nice paper on my Github. 
The idea is to package the quantum ADDER into a module that dynamically adjust based on the input size.
This dynamic adjustability is necessary to stack multiple ADDERs together for the creation of a MOD ADDER.


@author: Daniel Hutama
email: dhuta087@uottawa.ca
"""
import numpy as np

from qiskit import *

from qiskit import Aer, transpile

from qiskit.providers.aer import QasmSimulator
from qiskit.visualization import plot_histogram

def SUM():
    qSum = QuantumRegister(3)
    qSum_circ = QuantumCircuit(qSum, name = 'SUM')
    qSum_circ.cx(qSum[1], qSum[2])
    qSum_circ.cx(qSum[0], qSum[2])
    return qSum_circ.to_instruction()

def CARRY():
    qCarry = QuantumRegister(4)
    qCarry_circ = QuantumCircuit(qCarry, name = 'CARRY')
    qCarry_circ.ccx(qCarry[1], qCarry[2], qCarry[3])
    qCarry_circ.cx(qCarry[1], qCarry[2])
    qCarry_circ.ccx(qCarry[0], qCarry[2], qCarry[3])
    return qCarry_circ.to_instruction()

def rCARRY():
    qrCarry = QuantumRegister(4)
    qrCarry_circ = QuantumCircuit(qrCarry, name = 'rCARRY')
    qrCarry_circ.ccx(qrCarry[0], qrCarry[2], qrCarry[3])    
    qrCarry_circ.cx(qrCarry[1], qrCarry[2])
    qrCarry_circ.ccx(qrCarry[1], qrCarry[2], qrCarry[3])
    return qrCarry_circ.to_instruction()
    
SUM = SUM()
CARRY = CARRY()
rCARRY = rCARRY()


def get_binary_LSB_to_MSB(num, size):
    return np.binary_repr(num, size)[::-1]



def ADDER(n):
    #2n+1 qubit needed
    
    # initialize registers for addition gate
    A = QuantumRegister(n, 'a')
    B = QuantumRegister(n+1, 'b')
    C = QuantumRegister(n, 'c')
    QR_circ = QuantumCircuit(A, B, C, name='ADDER')
    


    # cascaded CARRY gates
    for i in range(n-1):
        QR_circ.append(CARRY, [C[i], A[i], B[i], C[i+1]]) 
    # final CARRY gate
    QR_circ.append(CARRY, [C[n-1], A[n-1], B[n-1], B[n]]) 
    # single CNOT in ADDER system
    QR_circ.cx(A[n-1], B[n-1])
    
    # cascaded SUM and rCARRY
    for i in range(n-1):
        QR_circ.append(SUM, [C[n-1-i], A[n-1-i], B[n-1-i]])
        QR_circ.append(rCARRY, [C[n-2-i], A[n-2-i], B[n-2-i], C[n-1-i]])
    # final SUM
    QR_circ.append(SUM, [C[0], A[0], B[0]])
    return QR_circ.to_instruction()

 



n = 3 #number of qubits in registers A and C, note register B will be of size n+1
ADDER = ADDER(n)
a = 5 #decimal value of register A
b = 3 #decimal value of register B


execute = 1
trials = 1000
classical = True
if execute == 1:


    # # initialize 10 qubit system
     A = QuantumRegister(n, 'a')        
     B = QuantumRegister(n+1, 'b')
     C = QuantumRegister(n, 'c')
    
    
     QR_circ = QuantumCircuit(A, B, C)
     c = QR_circ
     
     



     # initialize register A 
     a_bin = get_binary_LSB_to_MSB(a, n)
     for i in range(len(a_bin)):
         if a_bin[i] == '1':
             c.x(A[i])
    
     # initialize register B
     b_bin = get_binary_LSB_to_MSB(b, n)
     for i in range(len(b_bin)):
         if b_bin[i] == '1':
             c.x(B[i])
     

     enumerated_qbits = []
     # build list of target qbits for append function
     for i in range(n):
         enumerated_qbits.append(A[i])
     for i in range(n+1):
         enumerated_qbits.append(B[i])
     for i in range(n):
         enumerated_qbits.append(C[i])
     
     c.append(ADDER, enumerated_qbits)
     c.draw(fold=-1)
     
     c.measure_all()
    
     
     
     simulator = Aer.get_backend('aer_simulator')
     # simulator = QasmSimulator()
     c = transpile(c, simulator)
     result = simulator.run(c, shots = trials).result()
     counts = result.get_counts(c)
     out = list(counts.keys())[0][::-1]
    
     A_out = out[0:n]
     B_out = out[n: 2*n + 1]
     C_out = out[2*n+1::]

     print('IN  | Register A: {} ({}), Register B: {} ({})'.format(get_binary_LSB_to_MSB(a, n), a, get_binary_LSB_to_MSB(b, n+1), b))
     print('OUT | Register A: {} ({}), Register B: {} ({}), Register C: {}'.format(A_out, int(A_out[::-1], 2), B_out, int(B_out[::-1], 2), C_out))

c.draw(fold=-1)
