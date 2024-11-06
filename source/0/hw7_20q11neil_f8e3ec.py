# https://github.com/neilgupte75/CS_810_Quantum-Computing_Neil/blob/67697b0288014c72ab65277a04881c48c7351a19/hw7%20q11neil.py
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 18:38:23 2020

@author: Neil Gupte
"""

import numpy as np 
from qiskit import(QuantumCircuit,  execute,  Aer)
from qiskit.visualization import plot_histogram
# Use Aer's qasm_simulator
simulator = Aer.get_backend('qasm_simulator')
circuit1 = QuantumCircuit(2, 2)
circuit1.h(0)
 
circuit1.draw()
circuit1.x(1)

circuit1.cx(0, 1)
circuit1.draw()
circuit1.measure([0,1], [0,1])
job1 = execute(circuit1, simulator, shots=1000)
# Grab results from the job
result1 = job1.result()
# Returns counts
counts1 = result1.get_counts(circuit1)
print("\nTotal count for 10 and 01 are:",counts1)
plot_histogram(result1.get_counts(circuit1))