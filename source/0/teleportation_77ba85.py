# https://github.com/mariotoscano1997/Quantum-/blob/02154bdd3d5533182dda0409316ce0fdcf5d7e53/Teleportation.py
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 19:55:07 2022

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

def randomState():
    return random_statevector(2)

#Crea un circuito quantistico
qr=QuantumRegister(3,"q")
ac=QuantumRegister(1,"ancilla")
cr=ClassicalRegister(2,"c")
circuit = QuantumCircuit(qr,ac,cr)
print("prima")
initial=QuantumRegister(1,"init")
initial_gate = Initialize(randomState())
initial_gate_reverse=  initial_gate.gates_to_uncompute()
circuit=circuit.compose(initial_gate,[qr[0]])  
circuit.barrier()
#superimposizione
circuit.h(1)
circuit.cx(1,2)
psi=randomState()
circuit.barrier()
circuit.cx(0,1)
circuit.h(0)
circuit.barrier()
print("in mezzo")
print(qr[2])
circuit.measure(0,0)
circuit.measure(1,1)
circuit.x(qr[2]).c_if(cr[1],1)
circuit.z(qr[2]).c_if(cr[0],1)
print("fine")
print(qr[2])

circuit=circuit.compose(initial_gate_reverse,[qr[2]])
circuit.measure(qr[2], cr[0])

#Effettua la simulazione 
simulator=QasmSimulator()

compiled_circuit = transpile(circuit,simulator)
#Compilazione
job=simulator.run(compiled_circuit, shot=1000)

#ottiene i risulatati
results = job.result()
print(results.get_counts())    
counts=results.get_counts(circuit)    
#disegna il circuito
circuit.draw(output='mpl')
if list(counts.keys())[0][0]=='0':
    print("Fanculo")                
