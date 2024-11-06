# https://github.com/federico-rocco/VQE/blob/80424b81373ca4382e9449ce511f001eed279a5b/untitled3.py
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 18:38:09 2022

@author: cosmo
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 17:30:47 2022

@author: cosmo
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 19:22:37 2022

@author: cosmo
"""

from hamiltonian import Hamiltonian
from ansatz import ansatz
from solver import Eigensolver, State
from qiskit import Aer, IBMQ
from algorithm import Algorithm
from optimizer import Optimizer
from quarkonium import *
import matplotlib as mpt
import copy 
from qiskit.opflow import StateFn


coeffs = coeff()
hamiltonian = Hamiltonian(fermions,orbitals,coeffs)
ansatz = ansatz('quarkonium', n_fermions=fermions, n_qubits=orbitals)

options = {
    'shots':10**5,
    'ibmq':False,
    'seed':10,
    'backend':'qasm_simulator', #qasm/aer/ibmq_something
    #'device':'ibmq_athens' #to be simulated if using simulator
    }
algorithm = Algorithm(options)
optimizer = Optimizer('spsa', disp=False, max_iter=1000)
vqe = Eigensolver(fermions, orbitals, ansatz, hamiltonian(), optimizer, algorithm)
#print('my:',vqe.vqe_expval([4.61094861, 0.13099029]))
#print('qiskit:',vqe.vqe_qiskit())
params = [4.6106692 , 6.44062144]
aqc = vqe.get_ansatz(params)
operator = 0
for pauli in hamiltonian():
    operator += ~StateFn(pauli.pauli.to_pauli_op())@StateFn(aqc)
print('qiskit:', operator.eval(), 'my:', vqe.expval(params))
