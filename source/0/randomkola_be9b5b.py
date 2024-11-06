# https://github.com/nynazdravkovic/Quantum-tomography-of-random-entangled-states/blob/9442408599d35e012f8480e3ba8c56026fd623bc/randomKola.py
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 19:36:59 2020

@author: Nina
"""



import numpy as np
import itertools
from qiskit import QuantumRegister, QuantumCircuit,ClassicalRegister, execute, Aer
import json
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors import pauli_error
from qiskit.ignis.mitigation.measurement import (complete_meas_cal,CompleteMeasFitter)
import random
n=3
q=[]
c=[]
n_s=4
n_m=2
kolo=[]


def randomCirc(circ_list, n, n_single_qubit_gate, n_two_qubit_gate):
    niz=n_single_qubit_gate*[1] + n_two_qubit_gate*[2]
    random.shuffle(niz)
    for i in range(n_single_qubit_gate+n_two_qubit_gate):
        if niz[i]==1:
            qubit=random.randint(0, n-1)
            gate=random.randint(0,2)
            if gate==0:
                for circ in circ_list:
                    circ.h(qubit)
            if gate==1:
                for circ in circ_list:
                    circ.s(qubit)
            if gate==2:
                for circ in circ_list:
                    circ.t(qubit)
        if niz[i]==2:
            two_qubits=random.sample(range(0, n), 2)
            for circ in circ_list:
                circ.cx(two_qubits[0], two_qubits[1])



def get_noise(p):
    error_meas = pauli_error([('X',p), ('I', 1 - p)])
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(error_meas, "measure") # measurement error is applied to measurements        
    return noise_model



def parametri(N):   
    res =[ ele for ele in itertools.product([0,1,2,3], repeat = N)]
    return res

par=parametri(n)

def merX(kpom,i):
    kpom.h(i)
    return kpom

def merY(kpom,i):
    kpom.s(i)
    kpom.h(i)
    return kpom

def matricaZaGreske(brojQubita, noise_model):
    qr = QuantumRegister(brojQubita)
    meas_calibs, state_labels = complete_meas_cal(qr=qr, circlabel='mcal')
    backend = Aer.get_backend('qasm_simulator')
    job = execute(meas_calibs, backend=backend, shots=1000, noise_model=noise_model)
    cal_results = job.result()
    meas_fitter = CompleteMeasFitter(cal_results, state_labels, circlabel='mcal')
    return(meas_fitter.cal_matrix)


#PRAVLJENJE RANDOM KOLA

# sad jednostavno zelim da budu svi isti gejtovi ali on na neku foru izgubi registre i onda ih ne prepoznaje kasnije
for i in range(4**n):
    q.append(QuantumRegister(n))
    c.append(ClassicalRegister(n))
    kolo.append(QuantumCircuit(q[i],c[i]))
randomCirc(kolo,n,n_s,n_m)
for i in range(4**n):
    kolo[i].barrier()


#cuvam state vector
result = execute(kolo[0], Aer.get_backend('statevector_simulator')).result()
psi  = result.get_statevector(kolo[0])
psik=np.zeros((2**n,1),dtype=np.complex_)
for i in range(len(psi)):
    psik[i][0]=np.conj(psi[i])
#mat je matrica gustine koja je ocekivana
mat=np.outer(psi,psik)



#dodajemo merenja na svaki nacin
for i in range(4**n):
    for j in range(n):
        if par[i][j]==1:
            merX(kolo[i],j)
            kolo[i].measure(q[i][j],c[i][j])
        if par[i][j]==2:
            merY(kolo[i],j)
            kolo[i].measure(q[i][j],c[i][j])
        if par[i][j]==3:
            kolo[i].measure(q[i][j],c[i][j])
#izvrsavamo kola
job=[]
result=[]
counts=[]
noise_model = get_noise(0.15)
matricaSaSumom=[]



for i in range (4**n):
    results = execute( kolo[i], Aer.get_backend('qasm_simulator'),shots=1000, noise_model=noise_model).result().get_counts()
    counts.append(results)
    
    
matricaM = matricaZaGreske(n, noise_model)
 


#dict cuvam pomocu json-a
with open('random1countsp=0.15.txt', 'w') as outfile:
    json.dump(counts, outfile)
#np niz cuvam pomocu numpyja
np.save('random1matricaMp=0.15.npy', matricaM)
np.save('random1tacno.npy', mat)

    