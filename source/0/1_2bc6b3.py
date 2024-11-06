# https://github.com/Mirkesx/quantum_programming/blob/f7674cf833035a8115442a7f7ad49fef9f4c85ed/Exercises/1.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 17:25:59 2022

@author: mc
"""

import quantum_circuits as qcl
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.extensions import Initialize
from qiskit.providers.aer import QasmSimulator
import numpy as np
import random


def constGate():
    q1 = QuantumRegister(1)
    q2 = QuantumRegister(1)
    qc = QuantumCircuit(q1,q2)
    qc.x(q2)
    qc = qc.to_gate()
    qc.label = "Uf"
    return qc

def bilGate():
    q1 = QuantumRegister(1)
    q2 = QuantumRegister(1)
    qc = QuantumCircuit(q1,q2)
    qc.cx(q1, q2)
    qc = qc.to_gate()
    qc.label = "Uf"
    return qc

def constGateGen(n):
    q1 = QuantumRegister(n)
    q2 = QuantumRegister(1)
    qc = QuantumCircuit(q1,q2)
    qc.x(q2)
    qc = qc.to_gate()
    qc.label = "Uf"
    return qc

def bilGateGen(n):
    q1 = QuantumRegister(n)
    q2 = QuantumRegister(1)
    qc = QuantumCircuit(q1,q2)
    for i in range(n):
        if random.random() > 0.5:
            qc.cx(q1[i], q2)
    qc = qc.to_gate()
    qc.label = "Uf"
    return qc

def Deutch(gate=None):
    inp = QuantumRegister(1, "inp")
    out = QuantumRegister(1, "out")
    cr = ClassicalRegister(1, "cr")
    qc = QuantumCircuit(inp,out,cr)
    
    qc.x(out)
    
    qc.barrier()
    
    qc.h(inp)
    qc.h(out)
    
    qc.barrier()
    
    if gate == None:
        gate = constGate()
    qc = qc.compose(gate, list(inp) + list(out))
    
    
    qc.barrier()
    
    qc.h(inp)
    qc.barrier()
    
    qc.measure(inp,cr)
    qcl.draw_circuit(qc)
    
    counts = simulate(qc)
    print(counts)

def DeutchJosza(n = 2, gate=None):
    inp = QuantumRegister(n, "inp")
    out = QuantumRegister(1, "out")
    cr = ClassicalRegister(n, "cr")
    qc = QuantumCircuit(inp,out,cr)
    
    qc.x(out)
    
    qc.barrier()
    
    qc.h(inp)
    qc.h(out)
    
    qc.barrier()
    
    if gate == None:
        gate = constGateGen(n)
    qc = qc.compose(gate, list(inp) + list(out))
    
    
    qc.barrier()
    
    qc.h(inp)
    qc.barrier()
    
    qc.measure(inp,cr)
    qcl.draw_circuit(qc)
    
    counts = simulate(qc)
    print(counts)


    

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
    