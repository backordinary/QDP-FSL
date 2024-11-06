# https://github.com/3gaspo/guide-infoQ/blob/9f507df97667213c915dd8dfc5cc4290b264f2ee/codes/deustch.py
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 10:15:33 2020

@author: Gaspard
"""

from qiskit import QuantumCircuit, execute, Aer
from qiskit.visualization import plot_histogram
from math import sqrt
import numpy as np


def fonction_mystere(k):
    '''
    renvoie le circuit de la fonction fk
    '''
    qc = QuantumCircuit(2,2)
    if k == 0 :
        qc.cx(1,0)
    if k == 1 :
        qc.x(0)
        qc.cx(1,0)
    if k==3 :
        qc.x(0)
    return qc

def test(k_f,k_v):
    #k_f est l'entier qui commande la fonction mystère
    #k_v est la valeur à tester (O ou 1)
    backend = Aer.get_backend('qasm_simulator')
    
    qc = QuantumCircuit(2,2)
    
    if k_v == 1:
        qc.initialize([0,1],1)
    
    qcf = fonction_mystere(k_f)
    qcc = qc + qcf #on concatène les circuits
    #attention à l'ordre
    qcc.measure(1,1)
    qcc.measure(0,0)
    
    counts = execute(qcc, backend).result().get_counts()
    return(plot_histogram(counts))


#Deutsch
    
def deutsch(k_f):
    backend = Aer.get_backend('statevector_simulator')
    qc = QuantumCircuit(2,2)
    qc.x(0)
    qc.h(0)
    qc.x(1)
    qc.h(1)
    qc.barrier()
    
    qcc = qc + fonction_mystere(k_f)
    qcc.barrier()
    
    qcc.h(1)
    qcc.barrier()
    qcc.measure(1,0)
    return qcc

def resultat_deutsch(k_f):
    qc = deutsch(k_f)
    backend = Aer.get_backend('qasm_simulator')
    counts = execute(qc,backend).result().get_counts()
    if '01' in counts.keys():
        print('f est constante')
    else :
        print("f n'est pas constante")
    
    
