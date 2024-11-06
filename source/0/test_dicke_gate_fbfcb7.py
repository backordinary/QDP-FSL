# https://github.com/huub-d96/QA_Stage/blob/488891bdafdd72d800db0660a21a8cf759d5a7e4/test_Dicke_Gate.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 22 17:56:51 2021

@author: huub
"""
from math import pi, acos
from matplotlib import cm
import matplotlib.pyplot as plt
plt.interactive(True)
import time

from numpy import *
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import execute, Aer, IBMQ
from qiskit.providers.aer import QasmSimulator

def cry(theta):
    qc = QuantumCircuit(2)
  #  qc.cnot(0, 1)
  #  qc.ry(-theta/2, 1)
  #  qc.cnot(0, 1)
  #  qc.ry(theta / 2, 1)

    qc.ry((pi/2)-theta/2, 0)
    qc.cnot(1, 0)
    qc.ry(-((pi/2)-theta/2), 0)

    return qc.to_gate()


def ccry(theta):
    qc = QuantumCircuit(3)
    qc.toffoli(0, 1, 2)
    print(qc)
    qc.ry(-theta/2, 2)
    qc.toffoli(0, 1, 2)
    qc.ry(theta / 2, 2)
    return qc.to_gate()

def scs(n, k):
    qc = QuantumCircuit(n)
    qc.cnot(n - 2, n-1)
    theta = 2 * (acos(sqrt(1 / n)))
    qc.append(cry(theta), [n-2, n - 1])
    qc.cnot(n - 2, n-1)

    for m in range(k - 1):
        l = 2+m
        control = n-2-m
        qc.cnot(control-1, n-1)
        theta = 2 * (acos(sqrt((n-control) / n)))
        qc.append(ccry(theta), [n-1, control, control-1])

        qc.cnot(control-1, n-1)
    
    print(qc)    
    return qc.to_gate()


def dicke_init(n, k):
    #deterministic  Dicke state preparation (Bärtschi & Eidenbenz, 2019)
    #unoptimized version
    qc = QuantumCircuit(n)
    qc.x(range(n-k, n))
    for i in range(n, k, -1):
        qc.append(scs(i, k), range(0, i))

    for i in range(k, 1, -1):
        qc.append(scs(i, i-1), range(0, i))
        
    print(qc)    
    return qc.to_gate()

qc = QuantumCircuit(5, 5)
gate = dicke_init(5, 2)

qc.append(gate, range(0,5))