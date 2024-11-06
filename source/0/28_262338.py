# https://github.com/Mirkesx/quantum_programming/blob/d1bdcd14b5b7633541fc6d4ddad4e8fe9b1d6b47/Exercises/28.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 12:44:38 2022

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


def constGate(n):
    qc = QuantumCircuit(n+1)
    qc.x(n)
    qc = qc.to_gate()
    qc.label = "Const Gate"
    return qc


def balGate(n):
    qc = QuantumCircuit(n+1)
    for i in range(n):
        qc.cx(i,n)
    qc = qc.to_gate()
    qc.label = "Balanced Gate"
    return qc


def DJ(n, const = True):
    xr = QuantumRegister(n, "xr")
    yr = QuantumRegister(1, "yr")
    cr = ClassicalRegister(n, "cr")
    qc = QuantumCircuit(xr, yr, cr)
    
    qc.x(yr)
    qc.barrier()
    qc.h(xr)
    qc.h(yr)
    qc.barrier()
    if const:
        qc = qc.compose(constGate(n), qcl.get_qbits([xr,yr]))
    else:
        qc = qc.compose(balGate(n), qcl.get_qbits([xr,yr]))
    qc.barrier()
    qc.h(xr)
    qc.barrier()
    qc.measure(xr, cr)
    
    qcl.draw_circuit(qc)
    counts = qcl.simulate(qc, 1)
    print(counts)



def bvo(s):
    n = len(s)
    qc = QuantumCircuit(n+1)
    for i in range(n):
        if s[i] == "1":
            qc.cx(i,n)
    qc = qc.to_gate()
    qc.label = "BernsteinVazirani Gate"
    return qc


def BV(s):
    n = len(s)
    xr = QuantumRegister(n, "xr")
    yr = QuantumRegister(1, "yr")
    cr = ClassicalRegister(n, "cr")
    qc = QuantumCircuit(xr, yr, cr)
    
    qc.x(yr)
    qc.barrier()
    qc.h(xr)
    qc.h(yr)
    qc.barrier()
    qc = qc.compose(bvo(s), qcl.get_qbits([xr,yr]))
    qc.barrier()
    qc.h(xr)
    qc.barrier()
    qc.measure(xr, cr)
    
    qcl.draw_circuit(qc)
    counts = qcl.simulate(qc, 1)
    print(counts)




def so(s):
    n = len(s)
    s = s[::-1]
    idx = s.find("1")
    xr = QuantumRegister(n, "xr")
    yr = QuantumRegister(n, "yr")
    qc = QuantumCircuit(xr,yr)
    for i in range(n):
        qc.cx(xr[i], yr[i])
    for i in range(idx, n):
        qc.cx(xr[idx], yr[i])
    qc = qc.to_gate()
    qc.label = "Simon Gate"
    return qc


def Simon(s):
    n = len(s)
    xr = QuantumRegister(n, "xr")
    yr = QuantumRegister(n, "yr")
    cr = ClassicalRegister(n, "cr")
    qc = QuantumCircuit(xr, yr, cr)
    
    qc.h(xr)
    qc.barrier()
    qc = qc.compose(so(s), qcl.get_qbits([xr,yr]))
    qc.barrier()
    qc.measure(yr, cr)
    qc.barrier()
    qc.h(xr)
    qc.barrier()
    qc.measure(xr, cr)
    
    qcl.draw_circuit(qc)
    counts = qcl.simulate(qc, 1000)
    print(counts)




























