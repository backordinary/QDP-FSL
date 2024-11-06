# https://github.com/Mirkesx/quantum_programming/blob/56c600e86d40d80c3bd2c4d5d26a4daff8d7217c/Lessons/lesson_13.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 12:06:10 2021

@author: mc
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.extensions import Initialize
from qiskit.providers.aer import QasmSimulator
from qiskit.quantum_info import random_statevector
import quantum_circuits as qcl
import lesson_7 as qcircuit
import random

def q1_():
    qc = QuantumCircuit(2)
    return qc

def q1_0():
    qc = QuantumCircuit(2)
    qc.x(0)
    qc.cx(0,1)
    qc.x(0)
    return qc

def q1_0_1():
    qc = QuantumCircuit(2)
    qc.x(1)
    return qc

# oracle
Ui = qcircuit.CX.copy()
Uf = qcircuit.Idt(2)
Ut = np.kron(qcircuit.I, qcircuit.X)
Ux = np.array([
    [0,1,0,0],
    [1,0,0,0],
    [0,0,1,0],
    [0,0,0,1]
    ])


def Deutsch_alg(U):
    q = qcircuit.get_state('01')
    q = np.matmul(qcircuit.Had(2), q)
    q = np.matmul(U, q)
    q = np.matmul(np.kron(qcircuit.H, qcircuit.I), q)
    qcircuit.counts(q)
    
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



def Deutsch_qis(U):
    qc = QuantumCircuit(2,1)
    qc.x(1)
    qc.barrier()
    qc.h(0)
    qc.h(1)
    qc.barrier()
    qc = qc.compose(U, [0,1])
    qc.barrier()
    qc.h(0)
    qc.measure(0,0)
    qcl.draw_circuit(qc)
    counts = simulate(qc)
    print(counts)









