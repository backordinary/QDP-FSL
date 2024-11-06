# https://github.com/Mirkesx/quantum_programming/blob/f7674cf833035a8115442a7f7ad49fef9f4c85ed/Lessons/lesson_21.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 17:43:39 2022

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


def GeneralPhaseOracle(wl):
    n = len(wl[0])
    qc = QuantumCircuit(n)
    
    for sol in wl:
        for i in range(n):
            if sol[i] == "0":
                qc.x(i)
        qc.h(n-1)
        qc.mcx(list(range(n-1)), n-1)
        qc.h(n-1)
        # uncompute
        for i in range(n):
            if sol[i] == "0":
                qc.x(i)
        #qc.barrier()
    #qcl.draw_circuit(qc)
    return qc
    

def diffuser(n):
    s = '0'*n
    xr = QuantumRegister(n)
    qc = QuantumCircuit(xr)
    qc.h(xr)
    qc = qc.compose(GeneralPhaseOracle([s]))
    qc.h(xr)
    return qc

def GGroverOperator(wl):
    n = len(wl[0])
    qc = QuantumCircuit(n)
    
    # 2Pa - I
    gpo = GeneralPhaseOracle(wl)
    qc = qc.compose(gpo)
    
    # Refa
    diff = diffuser(n)
    qc = qc.compose(diff)
    #qcl.draw_circuit(qc)
    qc = qc.to_gate()
    qc.label = "Op Grov"
    return qc
    

def Grover(wl):
    gate = GGroverOperator(wl)
    n = len(wl[0])
    N =  2**n
    M = CountSolutions(gate, n, n)#len(wl)
    
    xr = QuantumRegister(n)
    cr = ClassicalRegister(n)
    qc = QuantumCircuit(xr, cr)
    qc.h(xr)
    
    it = int((np.pi/4) * np.sqrt(N / M))
    for i in range(it):
        qc = qc.compose(gate)
    
    for i in range(n):
        qc.measure(i, n-i-1)
    
    qcl.draw_circuit(qc)
    
    counts = qcl.simulate(qc)
    #print("Grover: {}".format())
    
def QFT(n):
    qc = QuantumCircuit(n)
    
    for t in range(n):
        qc.h(t)
        i = 1
        for c in range(t+1, n):
            phase = np.pi/(2**i)
            qc.cp(phase, c, t)
            i = i*2
        
    for q in range(int(n/2)):
        qc.swap(q, n-q-1)
    
    #qcl.draw_circuit(qc)
    return qc

T = lib.TGate()
S = lib.SGate()
Z = lib.ZGate()

def U(theta):
    return lib.U1Gate(theta)

    
    
def CountSolutions(gate, n, m=1):
    
    pr = QuantumRegister(n, "phase")
    gr = QuantumRegister(m, "gr")
    cr = ClassicalRegister(n, "cr")
    qc = QuantumCircuit(pr, gr, cr)
    
    qc.h(pr)
    qc.h(gr)
    #qc.barrier()
    cgate = gate.control()
    
    for i in range(n):
        for r in range(2**(n-i-1)):
            qc = qc.compose(cgate, [pr[i]]+[gr[q] for q in range(m)])
    
    
    #qc.barrier()
    iqft = QFT(n).inverse()
    iqft = iqft.to_gate()
    iqft.label = "QFT^"
    
    qc.append(iqft, pr)
    
    for q in range(n):
        qc.measure(q, n-q-1)
        
    qcl.draw_circuit(qc)
    
    counts = qcl.simulate(qc)
    print(counts)
    
    #counts = list(counts)
    #counts.sort(reverse=True)
    val = max(counts, key=counts.get)
    vmisurato = int(val, 2)
    theta = np.pi*(2*vmisurato)/2**n
    print("Rotation : {}".format(theta))
    N = 2**m
    M = N * np.sin(theta/2)**2
    print("Soluzioni {}".format(N-M))
    return N-M




















































