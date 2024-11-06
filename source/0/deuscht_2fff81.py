# https://github.com/Soula96/Quantumcomputing/blob/399c148863ff70f9cd933209f883ea92e6cf462a/Deuscht.py
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 16:57:42 2022

@author: MaxPr
"""
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.providers.aer import QasmSimulator
from qiskit.visualization import plot_histogram

def DeutschBlackBox(case):
    BlackBoxCircuit = QuantumCircuit(2)
    if (case == 'balanced'):
        b =  np.random.randint(1,4)
        b_str = format(b, '0'+str(2)+'b')
        for qubit in range(len(b_str)):
            if b_str[qubit] == '1':
                BlackBoxCircuit.x(qubit)
        BlackBoxCircuit.cx(0, 1)
        for qubit in range(len(b_str)):
            if b_str[qubit] == '1':
                BlackBoxCircuit.x(qubit)
    if (case == 'constant'):
        output = np.random.randint(2)
        #erzeuge zufällig die konstante Ausgabe, dies kann entweder 0 oder 1 sein
        if output == 1:
            BlackBoxCircuit.x(2) #flippt die Qubits x=[(01)(10)]
    BlackBoxGate = BlackBoxCircuit.to_gate()
    BlackBoxGate.name = "Black Box"
    BlackBoxCircuit.draw(output='mpl')
    return BlackBoxGate

BBGate = DeutschBlackBox('balanced')
DeutschCircuit = QuantumCircuit(2)
DeutschCircuit.h(0)
DeutschCircuit = DeutschBlackBox.to_gate()
DeutschCircuit.draw(output='mpl')

