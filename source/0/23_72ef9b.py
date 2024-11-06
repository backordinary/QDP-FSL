# https://github.com/Mirkesx/quantum_programming/blob/d1bdcd14b5b7633541fc6d4ddad4e8fe9b1d6b47/Exercises/23.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 15:54:10 2022

@author: mc
"""

import quantum_circuits as qcl
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.extensions import Initialize
from qiskit.providers.aer import QasmSimulator
from qiskit.quantum_info import random_statevector
import qiskit.circuit.library as lib
import numpy as np
import random
from fractions import Fraction


def GPO(wl):
    n = len(wl[0])
    mat = np.identity(2**n, int)
    for sol in wl:
        i = int(sol, 2)
        mat[i,i] = -1
    return mat

def get_state(state):
    n = len(state)
    N = 2**n
    new_state = np.array([0] * N)
    new_state[int(state, 2)] = 1
    return new_state

H = 1/np.sqrt(2) * np.array(
        [[1,1],
         [1,-1]]
    )

def Had(n):
    if n == 1:
        return H
    else:
        return np.kron(H, Had(n-1))



def Refa(a):
    N = len(a)
    return 2*np.matmul(np.transpose([a]), [a]) - np.identity(N, int)



def Grover(wl):
    n = len(wl[0])
    N = 2**n
    M = len(wl)
    t = int(np.floor(np.pi/4 * np.sqrt(N/M)))
    
    a = np.matmul(get_state("0"*n), Had(n))
    gpo = GPO(wl)
    refa = Refa(a)
    
    
    q0 = get_state("0" * n)
    print("q0: {}".format(q0))
    q1 = a.copy()
    print("q1: {}".format(q1))
    
    for i in range(t):
        print("Iterazione {}".format(i+1))
        q1 = np.matmul(gpo, q1)
        q1 = np.matmul(refa,q1)
        print("q1: {}".format(q1))
        print("------------------------------------")
    
    q2 = q1.copy()
    print("q2: {}".format(q2))
    
    






































