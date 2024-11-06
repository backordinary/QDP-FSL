# https://github.com/Mirkesx/quantum_programming/blob/f7674cf833035a8115442a7f7ad49fef9f4c85ed/Exercises/4.py
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
from qiskit.quantum_info import random_statevector
from qiskit.extensions import Initialize
import numpy as np
import random
import time

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


def entangledPair():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0,1)
    return qc


def Tp():
    psi = QuantumRegister(1, name="psi")    # Protocol uses 3 qubits
    ar = QuantumRegister(1, name="a")    # Protocol uses 3 qubits
    br = QuantumRegister(1, name="b")    # Protocol uses 3 qubits
    crz = ClassicalRegister(1, name="crz") # and 2 classical bits
    crx = ClassicalRegister(1, name="crx") # in 2 different registers
    cr0 = ClassicalRegister(1, name="cr0") # in 2 different registers
    qc = QuantumCircuit(psi, ar, br, crz, crx, cr0)
    
    init_gate = Initialize(random_statevector(2))
    init_gate.label = "init"
    inverse_init_gate = init_gate.gates_to_uncompute()
    inverse_init_gate.label = "disentangler"
    
    qc = qc.compose(init_gate, list(psi))    
    qc = qc.compose(entangledPair(), list(ar) + list(br))
    
    qc.barrier()
    
    qc.cx(psi,ar)
    qc.h(psi)
    
    qc.barrier()
    
    qc.measure(psi, crz)
    qc.measure(ar, crx)
    
    qc.barrier()
    
    qc.x(br).c_if(crx, 1)
    qc.z(br).c_if(crz, 1)
    qc.append(inverse_init_gate, list(br))
    
    qc.barrier()
    qc.measure(br, cr0)
    
    qc.barrier()
    qcl.draw_circuit(qc)
    
    #counts = simulate(qc)
    #print(counts)
    
    simulator = QasmSimulator()
    compiled_circuit = transpile(qc, simulator)
    shots = 1000
    job = simulator.run(compiled_circuit, shots=shots)
    result = job.result()
    counts = result.get_counts(compiled_circuit)
    
    count0 = 0
    count1 = 0
    for key in counts.keys():
        if key[0] == "0":
            count0 += counts[key]
        else:
            count1 += counts[key]
    print("Times the message was correctly received: {} \nTimes the message was not correctly received: {}".format(count0, count1))
    new_counts = qcl.reformat_counts(counts, shots)
    print(new_counts)
    

def SDC(m):
    m = m[::-1]
    ar = QuantumRegister(1, "ar")
    br = QuantumRegister(1, "br")
    cr = ClassicalRegister(2, "cr")
    qc = QuantumCircuit(ar, br, cr)
    
    
    qc.h(0)
    qc.cx(0,1)
    qc.barrier()
    
    if m[1] == "1":
        qc.x(ar)
    if m[0] == "1":
        qc.z(ar)
    
    qc.barrier()
    qc.cx(0,1)
    qc.h(0)
    qc.barrier()
        
    qc.measure(list(ar) + list(br), cr)
    
    qcl.draw_circuit(qc)
    counts = simulate(qc)
    print(counts)

SDC("10")
    