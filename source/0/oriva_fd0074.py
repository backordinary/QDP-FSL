# https://github.com/mariotoscano1997/Quantum-/blob/02154bdd3d5533182dda0409316ce0fdcf5d7e53/Oriva.py
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 11:25:44 2022

@author: tosca
"""

import numpy as np 
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.providers.aer import QasmSimulator
from qiskit.visualization import plot_histogram
from qiskit.tools.visualization import circuit_drawer
import matplotlib.pyplot as plt



#import quantum_circuits as qcl

#Crea un circuito quantistico
qr=QuantumRegister(2,"q")
ac=QuantumRegister(1,"ancilla")
cr=ClassicalRegister(2,"c")
circuit = QuantumCircuit(qr,ac,cr)
circuit.x(1)
circuit.x(0)
circuit.barrier()
circuit.ccx(qr[0], qr[1],ac[0])
circuit.cx(qr[0], qr[1])
circuit.barrier()
circuit.measure(ac[0], 0)
circuit.measure(qr[1], 1)

#Effettua la simulazione 
simulator=QasmSimulator()

compiled_circuit = transpile(circuit,simulator)
#Compilazione
job=simulator.run(compiled_circuit, shot=1000)

#ottiene i risulatati
results = job.result()
print(results.get_counts())        
#disegna il circuito
circuit.draw(output='mpl')
