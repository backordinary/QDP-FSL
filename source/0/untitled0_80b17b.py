# https://github.com/federico-rocco/VQE/blob/6bef802baa42b3033407cd7af7d604d4da69dbf2/untitled0.py
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 12:33:48 2022

@author: cosmo
"""

from hamiltonian import Hamiltonian
from ansatz import Ansatz
from solver import Eigensolver, State
from qiskit import Aer, IBMQ
from algorithm import Algorithm
from optimizer import Optimizer
from quarkoniumpaper import Quarkonium
import matplotlib
from matplotlib import pyplot as plt
from scipy.optimize import root
import numpy as np


fermions = 1
orbitals = 3
flavours = ['charmonium']
model = Quarkonium('charmonium')
coeffs = model.coeff(orbitals)
hamiltonian = Hamiltonian(fermions,orbitals,coeffs)
ansatz = Ansatz('quarkonium', n_fermions=fermions, n_qubits=orbitals, mp2=False)
optimizer = Optimizer('cobyla', disp=True, max_iter=1000)
options = {
    'shots':1024,
    'ibmq':False,
    'seed':1,
    'backend':'qasm_simulator', #qasm/aer/ibmq_something
    #'device':'ibm_nairobi' #to be simulated if using simulator
    }
algorithm = Algorithm(options)
    
vqe = Eigensolver(fermions, orbitals, ansatz, hamiltonian(), optimizer, algorithm)
optimized_parameters = vqe.optimize_parameters(vqe.analytic_expval)
analytic = vqe.analytic_expval(optimized_parameters)
print(analytic)

x=[]
y=[]
ran = np.logspace(3, 6, num=20)
for shots in ran:

    print('doing ', shots)
    options = {
        'shots':shots,
        'ibmq':False,
        'seed':1,
        'backend':'qasm_simulator', #qasm/aer/ibmq_something
        #'device':'ibm_nairobi' #to be simulated if using simulator
        }
    algorithm = Algorithm(options)
        
    vqe = Eigensolver(fermions, orbitals, ansatz, hamiltonian(), optimizer, algorithm)
    eigenvalue = vqe.expval(optimized_parameters)
    
    x.append(shots)
    y.append(eigenvalue)
    print(shots,eigenvalue)
    
plt.scatter(x,y, marker='.')
#plt.grid()
plt.xscale("log")
plt.axhline(y=analytic, color='black', linestyle='dashed')
plt.ylabel('Energy [MeV]')
plt.xlabel('Number of shots')
plt.show()