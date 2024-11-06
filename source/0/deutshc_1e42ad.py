# https://github.com/mariotoscano1997/Quantum-/blob/02154bdd3d5533182dda0409316ce0fdcf5d7e53/Deutshc.py
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 19:19:12 2022

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


Ui = Matrix.CX
Uf = Matrix.Idt(2)
Ut = np.kron(Matrix.I,Matrix.X)
Ux = np.array([[0,1,0,0],
              [1,0,0,0],
              [0,0,1,0],
              [0,0,0,1]])

def Deutsch(U):
    q=Matrix.get_state("01")
    q=np.matmul(Matrix.Had(2),q)
    q=np.matmul(U,q)
    q=np.matmul(np.kron(Matrix.H,Matrix.I),q)
    Matrix.counts(q)
    
def q1_():
    q = QuantumCircuit(2)
    return q
def q1_0():
    q = QuantumCircuit(2)
    q.x(0)
    q.cx(0,1)
    q.x(0)
    return q
def q1_1():
    q = QuantumCircuit(2)
    q.cx(0,1)
    return q
def q1_0_1():
    q = QuantumCircuit(2)
    q.x(1)
    return q

def QDeutshc(U):
    circuit= QuantumCircuit(2,1)
    circuit.x(1)
    circuit.barrier()
    circuit.h(0)
    circuit.h(1)
    circuit.barrier()

    circuit=circuit.compose(U,[0,1])
    circuit.barrier()

    circuit.h(0)
    circuit.barrier()

    circuit.measure(0,0)
    

    #Effettua la simulazione 
    simulator=QasmSimulator()

    compiled_circuit = transpile(circuit,simulator)
    #Compilazione
    job=simulator.run(compiled_circuit, shot=1000)

    #ottiene i risulatati
    results = job.result()
    print(results.get_counts())    
    counts=results.get_counts(circuit)    
    #disegna il circuito
    circuit.draw(output='mpl')
    print(counts)