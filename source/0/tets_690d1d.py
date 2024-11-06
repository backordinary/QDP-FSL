# https://github.com/Dheasra/TPIV---EPFL/blob/6d1f3dfa4eb35360b6447ea81c6c067b9f37e3ac/Grover%20TPIVa/tets.py
#exercise set 6 - Quantum Computation & Quantum Info
import math
import numpy as np
from numpy import pi
# importing Qiskit
from qiskit import QuantumCircuit, execute, Aer, IBMQ, ClassicalRegister, QuantumRegister
from qiskit.providers.ibmq import least_busy
from qiskit.providers.aer import QasmSimulator
from qiskit.tools.monitor import job_monitor
from qiskit.visualization import plot_histogram, plot_bloch_multivector
# %config InlineBackend.figure_format = 'svg' # Makes the images look nice
import matplotlib.pyplot as plt
import matplotlib

def mcz(crct, trgt, n, na):
    ntot = n + na
    #todo: remplacer n, na dans les arguments par Cqubit (liste des indices de qubit de contrôle)
    #todo: pour l'instant le code prend en contrôle tous les autres qubits que le target
    #Changing the last qubit with the target as the code works with the last qubit by default
    if trgt != 0:
        crct.swap(trgt, 0)
    if n > 2:
        #first ccnot gates
        crct.ccx(n-1,n-2, n)
        for i in range(n-2):
            crct.ccx(n-3-i,n+i, n+1+i)
        crct.cz(ntot-1,0) #controlled operation
        #second ccnot gates
        for i in range(n-2):
            crct.ccx(i,ntot-2-i, ntot-1-i)
        crct.ccx(n-1,n-2, n)
    if n==1:
        crct.z(0)
    if n==2:
        crct.cz(0,1)
    #Changing back the last qubit with the target
    if trgt != 0:
        crct.swap(trgt, 0)

N = 6 #nbr of qubits
Na = N-1 #nbr of ancilla qubits
Nc = N #nbr of classical bits

#Start of the circuit
test = QuantumCircuit(N+Na,Nc) #initializing

#Constructing the uniform superposition of states
for qubit in range(N):
    test.h(qubit)

test.barrier()

mcz(test,3,N,Na)


#visualization of the circuit
print(test) #in the terminal

#code snippet of oracle:
# #Universal oracles
# if trgt == 0:
#     for i in range(n):
#         crct.x(i)
#     mcz(crct, n, na)
#     for i in range(n):
#         crct.x(i)
# if trgt == 2**n-1:
#     mcz(crct, n, na)
# #Non universal oracles
# if n == 2:
#     if trgt == 0:
#         crct.z(0)
#         crct.z(1)
#         crct.cz(0,1)
#     if trgt == 1:
#         crct.x(1)
#         crct.cz(0,1)
#         crct.x(1)
#     if trgt == 2:
#         crct.x(0)
#         crct.cz(0,1)
#         crct.x(0)
#     if trgt == 3:
#         crct.cz(0,1)
# if n == 3:
#     if trgt == 0.6: #tragets = |000> and |101>
#         for i in range(n):
#             crct.z(i)
#         crct.cz(0,1)
#         crct.cz(1,2)
