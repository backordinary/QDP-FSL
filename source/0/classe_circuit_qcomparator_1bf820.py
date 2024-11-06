# https://github.com/valteresj2/Quantum-comparator-circuit/blob/b960fa6e9337b938eed7ee0086101756d07789da/classe_circuit_Qcomparator.py
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 13:04:10 2019

@author: valter.e.junior
"""

import numpy as np
from random import choices,sample
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import register
from qiskit import execute
import math


class comparator():
    def __init__(self,qubit=3,not_list=False):
        self.mu=0
        self.sigma=1
        self.x = np.random.gamma(self.mu, self.sigma, 2**qubit)
        self.sear=sample(list(self.x),1)
        self.n=np.int_(np.log(len(self.x))/np.log(2))
        self.N=2**self.n
        self.vet=[1/np.sqrt(self.N)]*self.N
        self.k=np.int_(math.acos(1/self.N)/math.acos((self.N-4+2)/(self.N)))
        
        self.vet=[]
        for i in range(self.N):
            valor_bina=bin(i).replace('b','0')
            if len(valor_bina)>self.n:
                valor_bina=valor_bina[(len(valor_bina)-self.n):len(valor_bina)]
            if len(valor_bina)<self.n:
                valor_bina='0'*(self.n-len(valor_bina))+valor_bina
            self.vet.append(valor_bina)
            
        
            
        valor_bina=bin(np.where(self.x==self.sear)[0][0]).replace('b','0')
        if len(valor_bina)>self.n:
            valor_bina=valor_bina[(len(valor_bina)-self.n):len(valor_bina)]
        if len(valor_bina)<self.n:
            valor_bina='0'*(self.n-len(valor_bina))+valor_bina
        self.vet_ref=valor_bina
        
        if not_list==True:
            self.vet.pop(np.where(np.array(self.vet)==self.vet_ref)[0][0])
        
    def create_circuit_comparator(self):
        qr = QuantumRegister(self.n)
        qref=QuantumRegister(self.n)
        qout = QuantumRegister(1)
        carry=QuantumRegister(1)
        cr = ClassicalRegister(1)
        
        qc = QuantumCircuit(qr, qref,qout,carry,cr,name='Grover')
        
        for valueref in self.vet:
        
            for j,valores in enumerate(valueref):
                if valores=='1':
                    qc.x(qr[j])
        
            for i,value in enumerate(self.vet_ref):
                if value=='1':
                    qc.x(qref[i])
                    
                    
            for t in range(0,self.n):    #XOR
                qc.cx(qr[t],qref[t])
        
            qc.x(qref) #NOT XOR
            
            if self.n==2:
                qc.ccx(qref[0],qref[1],qout[0])
                qc.reset(qr)
                qc.reset(qref)
                
            if self.n>=3:
                for k in range(0,self.n-2):
                    qc.cx(qref[k+1],carry[0])
                    qc.ccx(qref[k],qref[k+1],carry[0])
                    qc.cx(carry[0],qref[k+1])
                    qc.cx(carry[0],qref[k])
                    qc.reset(carry[0])     
                qc.ccx(qref[int(self.n-2)],qref[int(self.n-1)],qout[0])
                qc.reset(qr)
                qc.reset(qref)
                
        qc.h(qout[0])
        qc.x(qout[0])
        qc.u1(math.pi/16,qout[0])   
        qc.x(qout[0])   
        qc.h(qout[0])  
        
        qc.barrier(qout)
        qc.measure(qout[0], cr[0])
        
        return qc
    
    def generate_result(self):
        backend = 'local_qasm_simulator'
        qc=self.create_circuit_comparator()
        job_sim = execute(qc, backend, shots=1000)
        return print(job_sim.result().get_counts(qc))
        