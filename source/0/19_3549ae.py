# https://github.com/Mirkesx/quantum_programming/blob/f7674cf833035a8115442a7f7ad49fef9f4c85ed/Exercises/19.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 19:07:24 2022

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

def entangled_pair():
    qc = QuantumCircuit(2)
    
    qc.h(0)
    qc.cx(0,1)
    
    return qc


def entangled_three():
    qc = QuantumCircuit(3)
    
    qc.h(0)
    qc.cx(0,1)
    qc.cx(0,2)
    
    
    return qc


def get_state(s):
    n = len(s)
    N = 2**n
    state = np.array( [0] * N )
    i = int(s, 2)
    state[i] = 1
    return state


def Tp(state=None):
    if state == None:
        state = random_statevector(2)
    else:
        state = get_state(state)
    
    psi = QuantumRegister(1, "psi")
    alice = QuantumRegister(1, "alice")
    bob = QuantumRegister(1, "bob")
    cz = ClassicalRegister(1, "cz")
    cx = ClassicalRegister(1, "cx")
    cr = ClassicalRegister(1, "cr")
    qc = QuantumCircuit(psi, alice, bob, cz, cx, cr)
    
    init = Initialize(state)
    inverse_init = init.gates_to_uncompute()
    
    qc = qc.compose(init, psi)
    qc = qc.compose(entangled_pair(), qcl.get_qbits([alice, bob]))
    qc.barrier()
    qc.cx(psi,alice)
    qc.h(psi)
    qc.measure(psi,cz)
    qc.measure(alice,cx)
    qc.barrier()
    qc.x(bob).c_if(cx, 1)
    qc.z(bob).c_if(cz, 1)
    qc.barrier()
    qc = qc.compose(inverse_init, bob)
    qc.measure(bob, cr)
    
    qcl.draw_circuit(qc)
    counts = qcl.simulate(qc)
    print(counts)





def SC(bits="000"):
    qc = QuantumCircuit(3,3)
    
    qc = qc.compose(entangled_three())
    qc.barrier()
    if bits[2] == "1":
        qc.x(0)
    
    if bits[1] == "1":
        qc.z(0)
    
    if bits[0] == "1":
        qc.z(0)
    qc.barrier()
    qc = qc.compose(entangled_three().inverse())
    qc.barrier()
    
    qc.measure(0,2)
    qc.measure(1,1)
    qc.measure(2,0)
    
    qcl.draw_circuit(qc)
    counts = qcl.simulate(qc)
    print(counts)






















































