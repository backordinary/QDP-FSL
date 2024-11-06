# https://github.com/mariotoscano1997/Quantum-/blob/b1178175f7606f1c547865562488f39dd8610364/BV.py
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 10:40:20 2022

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
#ritorna una lista di indici in cui i bit sono 0
def bitnotset(x,n):
    x=binary(x,n)
    return [i  for i in range(n) if x[i]=='0']

#ritorna una lista di indici in cui i bit sono 1
def bitset(x,n):
    x=binary(x,n)
    return [i  for i in range(n) if x[i]=='1']

def cdot(x,y):
    n=len(x)
    r=0
    for i in range(n):
        r=r+int(x[i])*int(y[i])
    return r%2

def compile_and_run(circuit,  shots):
    simulator=QasmSimulator()
    compiled_circuit = transpile(circuit, simulator)
    job=simulator.run(compiled_circuit, shots=shots)
    results = job.result()
    counts= results.get_counts(compiled_circuit)
    return counts

def BVOracle(s):
    n=len(s)
    circuit  = QuantumCircuit(n+1)
    for x in range(2**n):
        if(cdot(binary(x,n),s)==1):
            for i in bitnotset(x, n):
                circuit.x(i)
            circuit.mcx(list(range(n)),n)
            for i in bitnotset(x, n):
                circuit.x(i)
            circuit.barrier()
    #circuit.draw(output='mpl')
    return circuit

def BVOracle2(s):
    n=len(s)
    xr=QuantumRegister(n, "x")
    ar=QuantumRegister(n, "anc")
    yr=QuantumRegister(1, "y")
    circuit  = QuantumCircuit(xr,ar,yr)
    
    for i in bitset(int(s,2),n):
        circuit.x(ar[i])
    for i in range(n):
        circuit.ccx(xr[i],ar[i], yr[0])
    #circuit.draw(output='mpl')
    return circuit


def BV(s):
    n=len(s)
    oracle=BVOracle(s)
    xr=QuantumRegister(n, "x")
    ar=QuantumRegister(n, "anc")
    yr=QuantumRegister(1, "y")
    cr=ClassicalRegister(n, "cr")
    circuit=QuantumCircuit(xr, ar, yr, cr)
    circuit.x(yr[0])
    circuit.barrier()
    for i in range(n):
        circuit.h(xr[i])
    circuit.h(yr[0])
    circuit.barrier()
    
    circuit=circuit.compose(oracle,list(range(n+1)))
    circuit.barrier()
    for i in range(n):
        circuit.h(xr[i])
    circuit.h(yr[0])
    circuit.barrier()
    for i in range(n):
        circuit.measure(i, n-i-1)
   
    counts=compile_and_run(circuit, 1) 
    r=list(counts.keys())[0]
    print("Stringa nascosta: ", r)
    if r==s:
        print("risultato corretto")
    else :
        print("Risultato non corretto")
    circuit.draw(output='mpl')
def BV2(s):
    n=len(s)
    oracle=BVOracle2(s)
    xr=QuantumRegister(n, "x")
    ar=QuantumRegister(n, "anc")
    yr=QuantumRegister(1, "y")
    cr=ClassicalRegister(n, "cr")
    circuit=QuantumCircuit(xr, ar, yr, cr)
    circuit.x(yr[0])
    circuit.barrier()
    for i in range(n):
        circuit.h(xr[i])
    circuit.h(yr[0])
    circuit.barrier()
    
    circuit=circuit.compose(oracle,list(range(2*n+1)))
    circuit.barrier()
    for i in range(n):
        circuit.h(xr[i])
    circuit.h(yr[0])
    circuit.barrier()
    for i in range(n):
        circuit.measure(i, n-i-1)
   
    counts=compile_and_run(circuit, 1) 
    r=list(counts.keys())[0]
    print("Stringa nascosta: ", r)
    if r==s:
        print("risultato corretto")
    else :
        print("Risultato non corretto")
    circuit.draw(output='mpl')
    
    