# https://github.com/Mirkesx/quantum_programming/blob/f7674cf833035a8115442a7f7ad49fef9f4c85ed/Exercises/13.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 18:50:18 2022

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


def FBI(value, n):
    qc = QuantumCircuit(n)
    
    qc.h(list(range(n)))
    for i in range(n):
        phase = value * np.pi / 2**(i)
        qc.p(phase, i)
        
    return qc


def QFT(n):
    qc = QuantumCircuit(n)
    
    for i in range(n):
        qc.h(i)
        t = 1
        for j in range(i+1,n):
            phase = np.pi / 2**t
            qc.cp(phase, j, i)
            t +=1
        
    
    for i in range(int(n/2)):
        qc.swap(i, n-i-1)
    
    return qc


def TestQFT(value, n):
    
    qc = QuantumCircuit(n, n)
    
    fbi = FBI(value, n)
    iqft = QFT(n).inverse()
    
    qc = qc.compose(fbi, list(range(n)))
    qc = qc.compose(iqft, list(range(n)))
    
    for i in range(n):
        qc.measure(i, n-i-1)
    
    qcl.draw_circuit(qc)
    
    counts = qcl.simulate(qc)
    qcl.print_counts(counts)
    

def U(theta):
    return lib.U1Gate(theta)

T = lib.TGate()
Z = lib.ZGate()
S = lib.SGate()
    
def QPE(gate, n_cont, m_cont = 1):
    cont = QuantumRegister(n_cont, "cont")
    q = QuantumRegister(m_cont, "q")
    cr = ClassicalRegister(n_cont, "cr")
    qc = QuantumCircuit(cont, q, cr)
    cgate = gate.control()
    
    qc.h(cont)
    qc.h(q)
    #qc.barrier()
    
    for i in range(n_cont):
        for j in range(2**(n_cont-i-1)):
            qc = qc.compose(cgate, [cont[i]] + [ q[k] for k in range(m_cont) ])
    
    iqft = QFT(n_cont).inverse()
    iqft = iqft.to_gate()
    iqft.label = "QFT^"
    
    qc = qc.compose(iqft, cont)
    
    for i in range(n_cont):
        qc.measure(cont[i], cr[n_cont-i-1])    
    
    #qcl.draw_circuit(qc)

    counts = qcl.simulate(qc)
    
    sorted_counts = qcl.sort_counts(counts)
    max_idx = list(sorted_counts)[0]
    estimated_phase = int(max_idx, 2) * (np.pi / 2**n_cont) / np.pi
    theta = estimated_phase * np.pi * 2
    print("Rotazione di {} pi".format(theta))
    N = 2**m_cont
    M = N * (np.sin(theta / (2)) ** 2)
    print("Soluzioni: {}".format(N-M))
    return N-M
    
def GPO(wl):
    n = len(wl[0])
    qc = QuantumCircuit(n)
    
    for sol in wl:
        for i in range(n):
            if sol[i] == '0':
                qc.x(i)
                
        qc.h(n-1)
        qc.mcx(list(range(n-1)), n-1)
        qc.h(n-1)
                
        for i in range(n):
            if sol[i] == '0':
                qc.x(i)
    
    #qcl.draw_circuit(qc)
    return qc

    
def diffuser(n):
    xr = QuantumRegister(n)
    qc = QuantumCircuit(xr)
    
    qc.h(xr)
    wl = "0"*n
    gpo = GPO([wl])
    qc = qc.compose(gpo, xr)    
    qc.h(xr)
    
    #qcl.draw_circuit(qc)
    return qc


def GrO(wl):
    n = len(wl[0])
    gpo = GPO(wl)
    diff = diffuser(n)
    qc = QuantumCircuit(n)
    qc = qc.compose(gpo)
    qc = qc.compose(diff)
    qc = qc.to_gate()
    qc.label = "gro"
    
    return qc
    
def Grover(wl):
    gro = GrO(wl)   
    n = len(wl[0])
    gr = QuantumRegister(n, "gr")
    cr = ClassicalRegister(n, "cr")
    qc = QuantumCircuit(gr, cr)  
    N = 2**n
    M = int(QPE(gro, n, n))
    t = int(np.floor((np.pi/4) * np.sqrt(N/M)))
    
    qc.h(gr)
    
    for i in range(t):
        qc = qc.compose(gro, gr)
    
    
    for i in range(n):
        qc.measure(i, n-i-1)
        
    qcl.draw_circuit(qc)
    counts = qcl.simulate(qc)
    sorted_counts = qcl.sort_counts(counts)
    qcl.print_counts(sorted_counts)
    



















    