# https://github.com/kosakhro/QAOA-for-exact-cover/blob/20d8c7762885dbcc7d6968f96a47dd20b7791673/qaoa_01.py
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 20:22:29 2020

@author: Arttu Huikuri
"""
from qiskit import *
from qiskit.visualization import plot_bloch_multivector
from qiskit.visualization import plot_histogram
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Making the quantum circuit
def circuitI(gamma, beta, J, h1, h2):
    qc = QuantumCircuit(2, 2)
    qc.h([0, 1])
    qc.h(1)
    qc.cz(0,1)
    qc.rx(2*gamma*J, 1)
    qc.cz(0,1)
    qc.h(1)
    qc.i(0)
    qc.rz(2*gamma*h1, 0)
    qc.rz(2*gamma*h2, 1)
    qc.rx(2*beta, 0)
    qc.rx(2*beta, 1)
    qc.measure([0,1], [0,1])
    return qc


simulator = Aer.get_backend('qasm_simulator')

# Initializing the variables
h1 = 0
h2 = 0
J = 1


#Setting the dimesions of the pasephase. The amount of simulated points = n**2
n = 61
p = np.pi/(n-1)
res = np.zeros([n, n, 4])

# Running a simulation for each point
for i in range(n):
    gamma = i*p
    for j in range(n):
        beta = j*p
        qc = circuitI(gamma, beta, J, h1, h2)
        counts = execute(qc, backend = simulator, shots=1000).result().get_counts()
        res[i][j][:] = np.array([int(counts.get('00', 0)), int(counts.get('01', 0)), int(counts.get('10', 0)),
                                 int(counts.get('11', 0))])


# Making the heatmap plots
sns.heatmap(res[:, :, 0])
plt.xlabel('\u03B2, [0,\u03C0]')
plt.ylabel('\u03B3, [0,\u03C0]')
plt.title('00')

plt.figure(2)
sns.heatmap(res[:, :, 1])
plt.xlabel('\u03B2, [0,\u03C0]')
plt.ylabel('\u03B3, [0,\u03C0]')
plt.title('01')

plt.figure(3)
sns.heatmap(res[:, :, 2])
plt.xlabel('\u03B2, [0,\u03C0]')
plt.ylabel('\u03B3, [0,\u03C0]')
plt.title('10')

plt.figure(4)
sns.heatmap(res[:, :, 3])
plt.xlabel('\u03B2, [0,\u03C0]')
plt.ylabel('\u03B3, [0,\u03C0]')
plt.title('11')

plt.figure(5)
sns.heatmap(res[:, :, 2]+ res[:, :, 1])
plt.xlabel('\u03B2, [0,\u03C0]')
plt.ylabel('\u03B3, [0,\u03C0]')
plt.title('11 + 01')
