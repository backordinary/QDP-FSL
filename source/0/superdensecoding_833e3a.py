# https://github.com/mariotoscano1997/Quantum-/blob/b6ded023e6af3a4dddb700389ca70c2fa9a72a11/SuperdenseCoding.py
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 20:42:12 2022

@author: tosca
"""

import Matrix
import numpy as np 
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.providers.aer import QasmSimulator
from qiskit.visualization import plot_histogram
from qiskit.tools.visualization import circuit_drawer
import matplotlib.pyplot as plt
from qiskit.quantum_info import random_statevector
from qiskit.extensions import Initialize
from random import randint


def create_bell_pair():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0,1)
    qc = qc.to_gate()
    qc.name = "BellPair"
    return qc

def encode_message(msg):
    qc = QuantumCircuit(1)
    if msg[1] == "1":
        qc.x(0)
    if msg[0] == "1":
        qc.z(0)
    qc = qc.to_gate()
    qc.name = "Encode"
    return qc

def decode_message():
    qc = QuantumCircuit(2)
    qc.cx(1,0)
    qc.h(1)
    qc = qc.to_gate()
    qc.name = "Decode"
    return qc

def DenseCoding(msg):
    if(len(msg)!=2  or msg[1] not in "01" or msg[0] not in "01" ):
        print("Messaggio non valido")
    qc = QuantumCircuit(2,2)
    qc = qc.compose(create_bell_pair(),[0,1])
    qc.barrier()
    qc = qc.compose(encode_message(msg),[0])
    qc.barrier()
    qc = qc.compose(decode_message(),[0,1])
    qc.measure(0,0)
    qc.measure(1,1)
    simulator=QasmSimulator()
    compiled_circuit = transpile(qc, simulator)
    job=simulator.run(compiled_circuit, shots=1000)
    results = job.result()
    counts= results.get_counts(qc)
    print("Recived: "+str(list(counts.keys())[0]))
    
    