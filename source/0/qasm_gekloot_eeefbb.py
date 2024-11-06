# https://github.com/klumel8/VQE/blob/b38c5b8625017c5c967ad262c7389fd184926a0c/Floris/qasm_gekloot.py
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 11:49:23 2020

@author: flori
"""

from qiskit import *
from math import pi
import numpy as np
from qiskit.visualization import plot_bloch_multivector, plot_histogram
import matplotlib
plt = matplotlib.pyplot 

N = 4
qc = QuantumCircuit(N,2)

#hamiltonian is of form
#H = h*sum(S_i(z)*S_(i+1)(z)) - j*sum(S_i)
J = 1
h = 1

#=== 3x3 periodic grid ===
#NN = np.array([[0,2],[0,6],[0,5],[0,1],[1,7],[1,2],[1,4],[2,8],[2,3],[5,3],[5,6],[5,4],[4,3],[4,7],[3,8],[6,8],[6,7],[7,8]]);

#=== 4x1 periodic grid ===
NN = np.array([[0,1], [1,2], [2,3], [3,0]])

for q in range(N):
    r = np.random.random_sample()*2
    qc.rx(pi*r,q)
qc.measure([2,1],[0,1])

print(qc.draw())

# Let's see the result
backend = Aer.get_backend('qasm_simulator')
data = execute(qc,backend,shots=1000)
results = data.result().get_counts()
print(results)
#lt.plot(final_state)
#plot_histogram(results)
#plt.show()
#fig.show()
