# https://github.com/JorgeAGR/nmsu-course-work/blob/6cd204abbc074734fb7e8ca0e693a15e1cbe4ede/PHYS520/HW5/set1.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 19:20:14 2020

@author: jorgeagr
"""

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, Aer, execute
from qiskit.tools.visualization import plot_histogram
import numpy as np

def XOR(input1, input2):
    
    q = QuantumRegister(2)
    c = ClassicalRegister(1)
    qc = QuantumCircuit(q, c)
    
    if input1 == '1':
        qc.x(q[0])
    if input2 == '1':
        qc.x(q[1])
    
    qc.cx(q[0], q[1])
    
    qc.measure(q[1],c[0])
    
    backend = Aer.get_backend('qasm_simulator')
    job = execute(qc,backend,shots=1,memory=True)
    output = job.result().get_memory()[0]
    
    return output

def AND(input1, input2):
    
    q = QuantumRegister(3)
    c = ClassicalRegister(1)
    qc = QuantumCircuit(q, c)
    
    if input1 == '1':
        qc.x(q[0])
    if input2 == '1':
        qc.x(q[1])
    
    qc.ccx(q[0], q[1], q[2])
    
    qc.measure(q[2],c[0])
    
    backend = Aer.get_backend('qasm_simulator')
    job = execute(qc,backend,shots=1,memory=True)
    output = job.result().get_memory()[0]
    
    return output

def NAND(input1, input2):
  
    q = QuantumRegister(3)
    c = ClassicalRegister(1)
    qc = QuantumCircuit(q, c)
    
    if input1 == '1':
        qc.x(q[0])
    if input2 == '1':
        qc.x(q[1])
    
    # Need state |x, y, 1>
    qc.x(q[2])
    qc.ccx(q[0], q[1], q[2])
    
    qc.measure(q[2],c[0])
    
    backend = Aer.get_backend('qasm_simulator')
    job = execute(qc,backend,shots=1,memory=True)
    output = job.result().get_memory()[0]
    
    return output

def OR(input1, input2):
  
    q = QuantumRegister(3)
    c = ClassicalRegister(1)
    qc = QuantumCircuit(q, c)
    
    if input1 == '1':
        qc.x(q[0])
    if input2 == '1':
        qc.x(q[1])
    
    qc.x(q[2])
    qc.x(q[0])
    qc.x(q[1])
    qc.ccx(q[0], q[1], q[2])
    
    qc.measure(q[2],c[0])
    
    backend = Aer.get_backend('qasm_simulator')
    job = execute(qc,backend,shots=1,memory=True)
    output = job.result().get_memory()[0]
    
    return output
    
print('\nResults for the XOR gate')
for input1 in ['0','1']:
    for input2 in ['0','1']:
        print('    Inputs',input1,input2,'give output',XOR(input1,input2))

print('\nResults for the AND gate')
for input1 in ['0','1']:
    for input2 in ['0','1']:
        print('    Inputs',input1,input2,'give output',AND(input1,input2))

print('\nResults for the NAND gate')
for input1 in ['0','1']:
    for input2 in ['0','1']:
        print('    Inputs',input1,input2,'give output',NAND(input1,input2))

print('\nResults for the OR gate')
for input1 in ['0','1']:
    for input2 in ['0','1']:
        print('    Inputs',input1,input2,'give output',OR(input1,input2))