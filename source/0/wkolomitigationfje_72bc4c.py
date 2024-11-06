# https://github.com/nynazdravkovic/Quantum-tomography-of-random-entangled-states/blob/9442408599d35e012f8480e3ba8c56026fd623bc/wKoloMitigationFje.py
import numpy as np
import itertools
from qiskit import QuantumRegister, QuantumCircuit,ClassicalRegister, execute, Aer
import math
import json
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors import pauli_error, depolarizing_error
from qiskit.ignis.mitigation.measurement import (complete_meas_cal,CompleteMeasFitter)

n=5
q=[]
c=[]
wKolo=[]

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
#Pravim matricu M
qr = QuantumRegister(5)
meas_calibs, state_labels = complete_meas_cal(qr=qr, circlabel='mcal')
noise_model = get_noise(0.1)
backend = Aer.get_backend('qasm_simulator')
job = execute(meas_calibs, backend=backend, shots=1000, noise_model=noise_model)
cal_results = job.result()
meas_fitter = CompleteMeasFitter(cal_results, state_labels, circlabel='mcal')
M=meas_fitter.cal_matrix
#Pravim Kolo
for i in range(4**n):
    q.append(QuantumRegister(n))
    c.append(ClassicalRegister(n))
    wKolo.append(QuantumCircuit(q[i],c[i]))
    #ovde stavljamo gejtove
    wKolo[i].x(q[i][1])
    wKolo[i].ry(1.37,q[i][3])
    wKolo[i].cx(q[i][3],q[i][1])
    wKolo[i].ry(0.955,q[i][0])
    wKolo[i].ry(math.pi/4,q[i][4])
    wKolo[i].cx(q[i][1],q[i][0])
    wKolo[i].cx(q[i][3],q[i][4])
    wKolo[i].ry(-0.955,q[i][0])
    wKolo[i].ry(-math.pi/4,q[i][4])
    wKolo[i].cx(q[i][0],q[i][1])
    wKolo[i].cx(q[i][4],q[i][3])
    wKolo[i].ry(math.pi/4,q[i][2])
    wKolo[i].cx(q[i][1],q[i][2])
    wKolo[i].ry(-math.pi/4,q[i][2])
    wKolo[i].cx(q[i][2],q[i][1])
    
    wKolo[i].barrier()

#dodajemo merenja na svaki nacin
for i in range(4**n):
    for j in range(n):
        if par[i][j]==1:
            merX(wKolo[i],j)
            wKolo[i].measure(q[i][j],c[i][j])
        if par[i][j]==2:
            merY(wKolo[i],j)
            wKolo[i].measure(q[i][j],c[i][j])
        if par[i][j]==3:
            wKolo[i].measure(q[i][j],c[i][j])
#izvrsavamo kola
backend2 = Aer.get_backend('qasm_simulator')
job=[]
result=[]
counts=[]
noise_model = get_noise(0.15)
matricaSaSumom=[]
for i in range (4**n):
    results = execute( wKolo[i], Aer.get_backend('qasm_simulator'), shots=10000, noise_model=noise_model).result().get_counts()
    meas_filter = meas_fitter.filter
    mitigated_results = meas_filter.apply(results)
    matricaSaSumom.append(mitigated_results)
#dict cuvam pomocu json-a
with open('matricaSaSumomp=0.15.txt', 'w') as outfile:
    json.dump(matricaSaSumom, outfile)




    
