# https://github.com/Mirkesx/quantum_programming/blob/d1bdcd14b5b7633541fc6d4ddad4e8fe9b1d6b47/Exercises/26.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 12:16:32 2022

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


def Tp(state = None):
    if state is None:
        state = random_statevector(2)
    
    psi = QuantumRegister(1, "psi")
    a = QuantumRegister(1, "alice")
    b = QuantumRegister(1, "bob")
    cz = ClassicalRegister(1, "cz")
    cx = ClassicalRegister(1, "cx")
    cr = ClassicalRegister(1, "cr")
    qc = QuantumCircuit(psi, a, b, cz, cx, cr)

    init = Initialize(state)
    inverse_init = init.gates_to_uncompute()
    
    qc = qc.compose(init, psi)
    qc.h(a)
    qc.cx(a,b)
    qc.barrier()
    qc.cx(psi,a)
    qc.h(psi)
    qc.measure(psi, cz)
    qc.measure(a, cx)
    qc.barrier()
    qc.z(b).c_if(cz,1)
    qc.x(b).c_if(cx,1)
    qc.barrier()
    qc = qc.compose(inverse_init, b)
    qc.measure(b, cr)
    
    qcl.draw_circuit(qc)
    counts = qcl.simulate(qc)
    print(counts)

def SC(bits = "00"):
    a = QuantumRegister(1, "alice")
    b = QuantumRegister(1, "bob")
    cr = ClassicalRegister(2, "cr")
    qc = QuantumCircuit(a, b, cr)
    
    qc.h(a)
    qc.cx(a,b)
    qc.barrier()
    if bits[0] == "1":
        qc.z(a)
    
    if bits[1] == "1":
        qc.x(a)
    
    qc.barrier()
    qc.cx(a,b)
    qc.h(a)
    qc.measure(a, cr[1])
    qc.measure(b, cr[0])
    
    qcl.draw_circuit(qc)
    counts = qcl.simulate(qc)
    print(counts)
    























