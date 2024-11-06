# https://github.com/Soula96/Quantumcomputing/blob/34fb7a36ee83d43de8d0930c5852911d7eb739dd/Test.py
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 13:19:16 2022

@author: MaxPr
"""
import numpy as np
from qiskit import *
import matplotlib
import matplotlib.pyplot as plt

circ = QuantumCircuit(1, 1)
circ.h(0)
circ.measure(range(1), range(1))

circ.draw(output='mpl')
print(circ)

backend_sim = Aer.get_backend('qasm_simulator')

job_sim = backend_sim.run(transpile(circ, backend_sim), shots=1000)
result_sim = job_sim.result()
counts = result_sim.get_counts(circ)
print(counts)

from qiskit.visualization import plot_histogram
plot_histogram(counts, (5,5), sort='asc')

