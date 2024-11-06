# https://github.com/Mirkesx/quantum_programming/blob/f7674cf833035a8115442a7f7ad49fef9f4c85ed/Lessons/lesson_18.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 17:47:09 2022

@author: mc
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.extensions import Initialize
from qiskit.providers.aer import QasmSimulator
from qiskit.quantum_info import random_statevector
from qiskit.quantum_info.operators import Operator
import quantum_circuits as qcl
import lesson_7 as qcircuit
import random

def Refq(n):
    s = [1]*(2**n)
    s = qcircuit.normalizza(s)
    s = np.array([list(s)])
    matrix = np.matmul(np.transpose(s), s)
    matrix = 2*matrix - qcircuit.Idt(n)
    return matrix
    #op = Operator(matrix)
    #return op
    
    
def PhaseOracle(w):
    n = len(w)
    P = qcircuit.Idt(int(n))
    i = int(w,2)
    P[i, i] = -1
    return P

def Grover(w):
    n = len(w)
    N = 2**n
    it = int((np.pi / 4) * np.sqrt(N))
    P = PhaseOracle(w)
    R = Refq(n)
    
    s = qcircuit.normalizza([1]*(N))
    for i in range(it):
        #print("it: {}".format(i))
        s = np.matmul(P,s)
        #print(s)
        s = np.matmul(R,s)
        #print(s)
        #print()
    
    print("it: {}".format(it))
    print(s)
    ind = int(w,2)
    print("{:.2f}%".format((s[ind]**2) * 100))

def PhaseOracleMultiple(wl):
    n = len(wl[0])
    P = qcircuit.Idt(int(n))
    for x in wl:
        i = int(x,2)
        P[i, i] = -1
    return P    
    
def GroverMultiple(wl):
    n = len(wl[0])
    M = len(wl)
    N = 2**n
    it = int((np.pi / 4) * np.sqrt(N/M))
    P = PhaseOracleMultiple(wl)
    R = Refq(n)
    
    s = qcircuit.normalizza([1]*(N))
    for i in range(it):
        s = np.matmul(P,s)
        s = np.matmul(R,s)
    
    print("it: {}".format(it))
    print(s)
    for w in wl:
        ind = int(w,2)
        if (s[ind]**2) * 100 > 0:
            print("{} - {:.2f}%".format(w, (s[ind]**2) * 100))