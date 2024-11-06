# https://github.com/mariotoscano1997/Quantum-/blob/77dada001417db7e09726f661ce60d8138afeef4/Groover.py
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 12:18:11 2022

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

#Operatore di riflessione
def Refq(s):
    n = len(s)
    s  = np.array([list(s)])
    matrix = np.matmul(np.transpose(s),s)
    matrix=2*matrix - Matrix.Idt(n)
    return matrix

def Refp(n):
    s=[1]*(2**n)    
    s=Matrix.normalizza(s)
    s  = np.array([list(s)])    
    m = np.matmul(np.transpose(s),s)
    matrix=(2*m) - Matrix.Idt(n)
    return matrix

def PhaseOracle(w):
    n = len(w)
    P  = Matrix.Idt(n)
    i = int(w,2)
    P[i,i]=-1
    return P

def PhaseOracle2(wl):
    n = len(wl[0])
    P  = Matrix.Idt(n)
    for w in wl:        
        i = int(w,2)
        P[i,i]=-1
    return P
#groover per un singolo elemento vincitore
def Groover(w):
    n = len(w)
    N = 2**n
    it = int((np.pi/4)*np.sqrt(N))
    P = PhaseOracle(w)
    R = Refp(n)
    print("it : ", (it))
    
    s= Matrix.normalizza([1]*N)
    for i in range(it):
        s=np.matmul(P,s)
        s=np.matmul(R,s)
        
    print(s)
    print("in")
    i = int(w,2)
    print((s[i]**2)*100, "%")
#per più vincitori senza conoscere m
def Groover2(wl):
    n = len(wl[0])
    N = 2**n
    it = int((np.pi/4)*np.sqrt(N))
    P = PhaseOracle2(wl)
    R = Refp(n)
    print("it : ", (it))
    
    s= Matrix.normalizza([1]*N)
    for i in range(it):
        s=np.matmul(P,s)
        print(s)
        s=np.matmul(R,s)
        print(s)
        
    print(s)
    print("in")
    i = int(wl[0],2)
    print((s[i]**2)*100, "%")
#groover per lista di vincitori conoscendo m 
def Groover3(wl):
    n = len(wl[0])
    N = 2**n
    m = len(wl)
    it = int((np.pi/4)*np.sqrt(N/m))
    P = PhaseOracle2(wl)
    R = Refp(n)
    print("it : ", (it))
    
    s= Matrix.normalizza([1]*N)
    for i in range(it):
        s=np.matmul(P,s)        
        s=np.matmul(R,s)      
        
    print(s)
    print("in")
    i = int(wl[0],2)
    print((s[i]**2)*100, "%")
    
    
    
