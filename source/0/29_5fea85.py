# https://github.com/Mirkesx/quantum_programming/blob/d1bdcd14b5b7633541fc6d4ddad4e8fe9b1d6b47/Exercises/29.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 12:56:20 2022

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




def FBI(value, n):
    qc = QuantumCircuit(n)
    
    for i in range(n):
        qc.h(i)
        theta = value * np.pi / 2**i
        qc.p(theta, i)
        
    qcl.draw_circuit(qc)
    return qc

def QFT(n):
    qc = QuantumCircuit(n)
    
    for i in range(n):
        qc.h(i)
        t = 1
        for j in range(i+1,n):
            theta = np.pi / 2**t
            qc.cp(theta, i, j)
            t += 1
    
    for i in range(int(n/2)):
        qc.swap(i, n-i-1)
    
    qcl.draw_circuit(qc)
    return qc


def TestQFT(value, n):
    
    qc = QuantumCircuit(n,n)
    
    qc = qc.compose(FBI(value,n))
    qc = qc.compose(QFT(n).inverse())
    
    for i in range(n):
        qc.measure(i, n-i-1)
    
    qcl.draw_circuit(qc)
    return qcl.simulate(qc)

T = lib.TGate()
S = lib.SGate()
Z = lib.ZGate()
Sdg = lib.SdgGate()

def U(theta):
    return lib.U1Gate(theta)



def QPE(gate, n, m=1):
    cgate = gate.control()
    xr = QuantumRegister(n, "cont")
    gr = QuantumRegister(m, "gr")
    cr = ClassicalRegister(n, "cr")
    qc = QuantumCircuit(xr, gr, cr)
    
    qc.h(xr)
    qc.x(gr)
    qc.barrier()
    
    for i in range(n):
        for j in range(2**(n-i-1)):
            qc = qc.compose(cgate, [xr[i]] + [gr[k] for k in range(m)])
        qc.barrier()
    
    iqft = QFT(n).inverse().to_gate()
    iqft.label = "QFT^"
    
    qc = qc.compose(iqft, xr)
    qc.barrier()
    
    for i in range(n):
        qc.measure(i, n-i-1)
    
    qcl.draw_circuit(qc)
    counts = qcl.simulate(qc)
    sorted_counts = qcl.sort_counts(counts)
    qcl.print_counts(sorted_counts)
    
    value = int( list(sorted_counts.keys())[0],2)
    theta = value * 2 / 2**n
    print("Theta {} pi".format(theta))


def GPO(wl):
    n = len(wl[0])
    qc = QuantumCircuit(n)
    for sol in wl:
        for i in range(n):
            if sol[i] == "0":
                qc.x(i)
        qc.h(n-1)
        qc.mcx(list(range(n-1)),n-1)
        qc.h(n-1)
        for i in range(n):
            if sol[i] == "0":
                qc.x(i)
    return qc

def diffuser(n):
    sol = "0"*n
    wl = [sol]
    qc = QuantumCircuit(n)
    qc.h(list(range(n)))
    qc = qc.compose(GPO(wl))
    qc.h(list(range(n)))
    return qc

def Gro(wl):
    n = len(wl[0])
    qc = QuantumCircuit(n)
    qc = qc.compose(GPO(wl))
    qc = qc.compose(diffuser(n))
    qc = qc.to_gate()
    qc.label = "Grover"
    return qc
    


def CountSolutions(gate, n, m):
    cgate = gate.control()
    xr = QuantumRegister(n, "cont")
    gr = QuantumRegister(m, "gr")
    cr = ClassicalRegister(n, "cr")
    qc = QuantumCircuit(xr, gr, cr)
    
    qc.h(xr)
    qc.h(gr)
    qc.barrier()
    
    for i in range(n):
        for j in range(2**(n-i-1)):
            qc = qc.compose(cgate, [xr[i]] + [gr[k] for k in range(m)])
        qc.barrier()
    
    iqft = QFT(n).inverse().to_gate()
    iqft.label = "QFT^"
    
    qc = qc.compose(iqft, xr)
    qc.barrier()
    
    for i in range(n):
        qc.measure(i, n-i-1)
    
    qcl.draw_circuit(qc)
    counts = qcl.simulate(qc)
    sorted_counts = qcl.sort_counts(counts)
    #qcl.print_counts(sorted_counts)
    
    value = int( list(sorted_counts.keys())[0],2)
    theta = value * 2 / 2**n
    print("Theta {} pi".format(theta))
    N = 2**m
    M = N * np.sin(theta * np.pi /2 )**2
    print("Soluzioni: {}".format(N-M))
    return N-M


def Grover(wl):
    n = len(wl[0])
    gate = Gro(wl)
    xr = QuantumRegister(n)
    cr = ClassicalRegister(n)
    qc = QuantumCircuit(xr, cr)
    
    N = 2**n
    #M = len(wl)
    M = CountSolutions(gate, n, n)
    t = int( (np.pi/4) * np.sqrt(N/M) )
    
    qc.h(xr)
    qc.barrier()
    for i in range(t):
        qc = qc.compose(gate, xr)
    qc.barrier()
    
    qc.measure(xr, cr)
    qcl.draw_circuit(qc)
    counts = qcl.simulate(qc)
    qcl.print_counts(qcl.sort_counts(counts))













































