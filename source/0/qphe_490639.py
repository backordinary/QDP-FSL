# https://github.com/amartinfer/QPhE-QiskitCampEurope/blob/95b286fd047ffc62a2d371dafbdcee9e97ccaca7/QPhE.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 13:34:04 2019

@author: ana
"""

import numpy as np
from numpy import linalg as LA
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import BasicAer, execute
from qiskit.tools.visualization import plot_histogram
#from qiskit.tools.monitor import job_monitor
#from qiskit.providers.ibmq import least_busy
#import math
from scipy.linalg import expm
import qiskit

A=np.array([[0.2,0.4],[0.4,0.8]])
Ham=1/(np.matrix.trace(A))*A
l,u=LA.eig(Ham)
UH=expm(1j*Ham)
#state_vector=[u[:1]]
state_vector=[u[0][j] for j in range(2)]
#print(state_vector)
(th, ph, lam)=qiskit.quantum_info.synthesis.two_qubit_decompose.euler_angles_1q(UH)

q=QuantumRegister(2)
c=ClassicalRegister(2)
qc=QuantumCircuit(q,c)

qc.initialize(state_vector,[q[1]])
#preparation of the three first qubits (3-bit eigenvalue estimation)
qc.h(q[0])


#Controlled U_H
qc.cu3(th, ph, lam, q[0],q[1])

#Inverse QFT
qc.h(q[0])

#projection and meassurement
qc.barrier(q[0])
qc.barrier(q[1])
qc.measure(q[0],c[0])
qc.measure(q[1],c[1])

#Run on qasm simulator
backend_qasm=BasicAer.get_backend('qasm_simulator')
shot=8192
job_qasm=execute(qc,backend_qasm,shots=shot)
result_qasm=job_qasm.result()
counts=result_qasm.get_counts(qc)
print(counts)
Emo1=(counts.get('11'))
Emo2=(counts.get('10'))
Emo3=(counts.get('01'))
Emo4=(counts.get('00'))
ProbEmo1=round(Emo1/shot,2)
ProbEmo2=round(Emo2/shot,2)
ProbEmo3=round(Emo3/shot,2)
ProbEmo4=round(Emo4/shot,2)
plot_histogram(counts)
probs = {"11": ProbEmo1, "10": ProbEmo2, "01": ProbEmo3, "00": ProbEmo4}
print(probs)
