# https://github.com/Mirkesx/quantum_programming/blob/f7674cf833035a8115442a7f7ad49fef9f4c85ed/Exercises/3.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 18:02:48 2022

@author: mc
"""

import quantum_circuits as qcl
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.extensions import Initialize
from qiskit.providers.aer import QasmSimulator
from qiskit_textbook.tools import simon_oracle
import numpy as np
import random
import time


def genSimon(s):
    #qcc = simon_oracle(s)
    #qcl.draw_circuit(qcc)
    s = s[::-1]
    n = len(s)
    
    q1 = QuantumRegister(n,"q1")
    q2 = QuantumRegister(n,"q2")
    qc = QuantumCircuit(q1,q2)
    
    for i in range(n):
        qc.cx(q1[i],q2[i])
        
    index = s.find("1")
    
    for i in range(index, n):
        if s[i] == "1":
            qc.cx(q1[index],q2[i])
        
    #qcl.draw_circuit(qc)
    
    #qcc = simon_oracle(s)
    #qcl.draw_circuit(qcc)
    qc = qc.to_gate()
    qc.label = "Uf"
    
    return qc


def Simon(s):
    n = len(s)
    q1 = QuantumRegister(n,"inp")
    q2 = QuantumRegister(n, "out")
    cr = ClassicalRegister(n, "cr")
    qc = QuantumCircuit(q1, q2, cr)
    
    qc.h(q1)
    qc.barrier()
    
    gate = genSimon(s)
    qc = qc.compose(gate, list(q1) + list(q2))
    
    qc.barrier()
    
    qc.measure(q2,cr)
    
    qc.barrier()
    
    qc.h(q1)
    
    qc.measure(q1, cr)
    
    qcl.draw_circuit(qc)
    
    counts = simulate(qc)
    print(counts)
    bdots(counts, s)
    
def bdots(counts, s):
    for key in counts.keys():
        acc = 0
        for i in range(len(key)):
            acc += int(s[i]) * int(key[i])
        acc = acc%2
        print("{} . {} = {} (mod 2)".format(s, key, acc))

def simulate(qc):
    simulator = QasmSimulator()
    compiled_circuit = transpile(qc, simulator)
    shots = 1000
    job = simulator.run(compiled_circuit, shots=shots)
    result = job.result()
    #qcl.draw_circuit(qc)
    counts = result.get_counts(compiled_circuit)
    #new_counts = qcl.reformat_counts(counts, shots)
    #return new_counts
    return counts
    return int(list(counts.keys())[0])    