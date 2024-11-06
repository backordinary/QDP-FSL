# https://github.com/mariotoscano1997/Quantum-/blob/b6ded023e6af3a4dddb700389ca70c2fa9a72a11/Simon.py
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 10 12:57:45 2022

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

#prende un intero x, ritorna la sua "forma" binaria
def binary(x,n):
    formatstr ="{:0>"+str(n)+"b}"
    x= formatstr.format(x)
    return x
def xor(a,b):
    if a==b :
        return 0
    return 1
#ritorna una lista di indici in cui i bit sono 0
def bitnotset(x,n):
    x=binary(x,n)
    return [i  for i in range(n) if x[i]=='0']

#ritorna una lista di indici in cui i bit sono 1
def bitset(x,n):
    x=binary(x,n)
    return [i  for i in range(n) if x[i]=='1']

def compile_and_run(circuit,  shots):
    simulator=QasmSimulator()
    compiled_circuit = transpile(circuit, simulator)
    job=simulator.run(compiled_circuit, shots=shots)
    results = job.result()
    counts= results.get_counts(compiled_circuit)
    return counts


def strxor(x,hs):
    r=""
    n=len(hs)
    x=binary(x,n)
    for i in range(n):
        r=r+str(xor(x[i], hs[i]))
    return r

def SimonOracle(hs):
    n  = len(hs)
    sol = [None]*(2**n)
    out = 1
    for x in range(2**n):
        if(sol[x]==None):
            sol[x] = out #x
            y = int(strxor(x,hs),2)
            sol[y]=out  #y=x xor hs
            out = out + 1
    print(sol)
    qc = QuantumCircuit(2*n)
    for x in range(2*n):
        y=sol[x]
        for i in bitnotset(x, n):
            qc.x(i)
        for i in bitset(y,n):
            qc.mcx(list(range(n)),n+i)
        for i in bitnotset(x, n):
            qc.x(i)
        qc.barrier()
    qc.draw(output='mpl')
    return qc
def cdot(x,y):
    n=len(x)
    r=0
    for i in range(n):
        r=r+int(x[i])*int(y[i])
    return r%2
def SimonQ(hs):
    n=len(hs)
    xr= QuantumRegister(n, "x")
    yr= QuantumRegister(n, "y")
    cr= ClassicalRegister(n, "cr")
    qc=QuantumCircuit(xr, yr, cr)
    qc.h(xr)
    qc.barrier()
    oracle =SimonOracle(hs)
    qc=qc.compose(oracle, list(range(2*n)))
    qc.barrier()
    qc.measure(yr, cr)  
    qc.h(xr)
    qc.measure(xr, cr)      
    counts=compile_and_run(qc, 1);
    z=list(counts.keys())[0]
    if(cdot(z,hs)==0):
        print("Risultato corretto")
    else :
        print("Risultato non corretto")
        
    return z
def Simon(hs):
    n = len(hs)
    S = [None]*(2**n)
    nsol = 0
    iterazioni=0
    while(nsol <n-1):
        z=SimonQ(hs)
        iterazioni = iterazioni+1
        if((z!="0"*n) and S[int(z,2)]==None):
            S[int(z,2)]=1
            nsol=nsol+1
    print("Soluzione trovata in ", iterazioni, "Iterazioni")