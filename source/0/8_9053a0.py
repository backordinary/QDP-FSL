# https://github.com/Mirkesx/quantum_programming/blob/f7674cf833035a8115442a7f7ad49fef9f4c85ed/Exercises/8.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 11:02:50 2022

@author: mc
"""

import quantum_circuits as qcl
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.extensions import Initialize
from qiskit.providers.aer import QasmSimulator
from qiskit.quantum_info import random_statevector
import numpy as np
import random


def random_state():
    return random_statevector(2)

def bell_pair():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0,1)
    return qc
    gate = qc.to_gate()
    gate.label = "B00"
    return gate


def QT(state=None):
    if state is None:
        state = random_state()
        
    psi = QuantumRegister(1, "psi")
    ar = QuantumRegister(1, "alice")
    br = QuantumRegister(1, "bob")
    cx = ClassicalRegister(1, "cx")
    cz = ClassicalRegister(1, "cz")
    cr = ClassicalRegister(1, "cr")
    
    qc = QuantumCircuit(psi, ar, br, cz, cx, cr)
    
    # Initialize psi
    qc.barrier()
    init_gate = Initialize(state)
    init_gate.label = "init"
    inverse_init_gate = init_gate.gates_to_uncompute()
    qc.append(init_gate, psi)
    
    # Make entangled pair
    qc.barrier()
    qc = qc.compose(bell_pair(), qcl.get_qbits([ar, br]))
    
    # Alice prepares her qbits
    qc.barrier()
    qc.cx(psi, ar)
    qc.h(psi)
    
    # Alice measures her qbits
    qc.barrier()
    qc.measure(psi, cz)
    qc.measure(ar, cx)
    
    # Bob applies X or Z operators
    qc.barrier()
    qc.x(br).c_if(cx, 1)
    qc.z(br).c_if(cz, 1)
    
    # Bob works with his qbit
    qc.barrier()
    qc.append(inverse_init_gate, br)
    qc.measure(br, cr)
    
    
    qcl.draw_circuit(qc)
    
    counts = qcl.simulate(qc)
    count0 = 0
    count1 = 0
    for key in counts.keys():
        if key[0] == "0":
            count0 += counts[key]
        else:
            count1 += counts[key]
    print("Times the message was correctly received: {} \nTimes the message was not correctly received: {}\n{}".format(count0, count1, counts))

def SC(bits="00"):
    ar = QuantumRegister(1, "alice")
    br = QuantumRegister(1, "bob")
    cr = ClassicalRegister(2, "result")
    qc = QuantumCircuit(ar, br, cr)
    
    # Make entangled pair
    qc.barrier()
    qc.h(ar)
    qc.cx(ar, br)
    
    # Alice prepares her qbit
    qc.barrier()
    if bits[1] == "1":
        qc.x(ar)
    if bits[0] == "1":
        qc.z(ar)
    
    # Bob performs disentanglement and measures
    qc.barrier()
    qc.cx(ar, br)
    qc.h(ar)
    
    qc.barrier()
    qc.measure(ar, cr[1])
    qc.measure(br, cr[0])
    
    qcl.draw_circuit(qc)
    counts = qcl.simulate(qc)
    
    print(counts)










