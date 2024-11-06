# https://github.com/SaulPuente/QuGamers/blob/c367ec78598834a2600d51657ec0a6dbfb1e18c7/QuantumEngine.py
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 03:14:30 2021

@author: saulp
"""

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile, assemble, Aer, IBMQ, execute
from qiskit.visualization import plot_bloch_multivector, plot_histogram
from qiskit.quantum_info.states import Statevector, partial_trace
from numpy.random import choice as choice
#import numpy as np

class qubit():
    def __init__(self, ID):
        self.qr = QuantumRegister(1, 'q' + str(ID))
        #self.cr = ClassicalRegister(1, 'c' + str(ID))
        
class circuit():
    def __init__(self):
        self.qc = QuantumCircuit()
        
    def addQubit(self,qr):#,cr):
        self.qc.add_register(qr)#,cr)
        
    def H(self,qr):
        self.qc.h(qr)
    
    def X(self,qr):
        self.qc.x(qr)
    
    def Y(self,qr):
        self.qc.y(qr)
    
    def Z(self,qr):
        self.qc.z(qr)
        
    def R(self,qr,angle):
        self.qc.ry(angle,qr)
    
    def CX(self,control,target):
        self.qc.cx(control,target)
        
    def collapse(self,soldier,gs):
        
        v = choice([0,1],1,[soldier.status["prob0"],soldier.status["prob1"]])
        if v == 0:
            gs.board[soldier.status["state1"][0]][soldier.status["state1"][1]] = "--"
            if gs.board[soldier.status["istate"][0]][soldier.status["istate"][1]] == "--":
                soldier.status["state1"] = soldier.status["istate"]
            else:
               soldier.status["state1"] = ("","") 
        elif v == 1:
            gs.board[soldier.status["state0"][0]][soldier.status["state0"][1]] = "--"
            soldier.status["state0"] = soldier.status["state1"]
            if gs.board[soldier.status["istate"][0]][soldier.status["istate"][1]] == "--":
                soldier.status["state1"] = soldier.status["istate"]
            else:
               soldier.status["state1"] = ("","") 
        else:
            print("error")
            
        soldier.status["superposition"] = False
        self.qc.reset(soldier.qubit.qr)#
        return v

def get_probs(qc,ID,n):
    p0 = 0
    p1 = 0
    
    sv = Statevector.from_instruction(qc)
    probs = sv.probabilities()
    
    p = 1
    
    for i in range(2**n):
        if i in range(0,2**n,(2**n)//(2**(n-ID))):
            if p == 1:
                p = 0
            else:
                p = 1
        if p == 0:
            p0 += probs[i]
        else:
            p1 += probs[i]
    return round(p0,3), round(p1,3)