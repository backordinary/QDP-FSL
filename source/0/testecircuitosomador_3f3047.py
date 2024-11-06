# https://github.com/asgunzi/3qubitAdderQiskit/blob/2bb1269af4f5cf4130515464ec8d9900f684e7ca/TesteCircuitoSomador.py
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 11:24:40 2020

@author: asgun
"""

from qiskit import *

qin = QuantumRegister(9)
qsum = QuantumRegister(3)

cout = ClassicalRegister(3)

qc = QuantumCircuit(qin, qsum, cout)

 

#Inicialização 
# se tem x, inicializa com 1
qc.x(qin[0])
# qc.x(qin[1])
qc.x(qin[2])
qc.x(qin[3])
# qc.x(qin[4])
qc.x(qin[5])
qc.x(qin[6])
qc.x(qin[7])
qc.x(qin[8])
qc.cx(qin[0],qsum[0])

if 0>1 and 0<8:
    qc.mct([qin[0+1],qsum[0],qsum[1]], qsum[2])

if 0<8:
    qc.ccx(qin[0+1],qsum[0],qsum[1])
qc.cx(qin[1],qsum[0])

if 1>1 and 1<8:
    qc.mct([qin[1+1],qsum[0],qsum[1]], qsum[2])

if 1<8:
    qc.ccx(qin[1+1],qsum[0],qsum[1])
qc.cx(qin[2],qsum[0])

if 2>1 and 2<8:
    qc.mct([qin[2+1],qsum[0],qsum[1]], qsum[2])

if 2<8:
    qc.ccx(qin[2+1],qsum[0],qsum[1])
qc.cx(qin[3],qsum[0])

if 3>1 and 3<8:
    qc.mct([qin[3+1],qsum[0],qsum[1]], qsum[2])

if 3<8:
    qc.ccx(qin[3+1],qsum[0],qsum[1])
qc.cx(qin[4],qsum[0])

if 4>1 and 4<8:
    qc.mct([qin[4+1],qsum[0],qsum[1]], qsum[2])

if 4<8:
    qc.ccx(qin[4+1],qsum[0],qsum[1])
qc.cx(qin[5],qsum[0])

if 5>1 and 5<8:
    qc.mct([qin[5+1],qsum[0],qsum[1]], qsum[2])

if 5<8:
    qc.ccx(qin[5+1],qsum[0],qsum[1])
qc.cx(qin[6],qsum[0])

if 6>1 and 6<8:
    qc.mct([qin[6+1],qsum[0],qsum[1]], qsum[2])

if 6<8:
    qc.ccx(qin[6+1],qsum[0],qsum[1])
qc.cx(qin[7],qsum[0])

if 7>1 and 7<8:
    qc.mct([qin[7+1],qsum[0],qsum[1]], qsum[2])

if 7<8:
    qc.ccx(qin[7+1],qsum[0],qsum[1])
qc.cx(qin[8],qsum[0])

if 8>1 and 8<8:
    qc.mct([qin[8+1],qsum[0],qsum[1]], qsum[2])

if 8<8:
    qc.ccx(qin[8+1],qsum[0],qsum[1])

qc.measure(qsum[:],cout[:])
        
#print(qc)
qc.draw(output = 'mpl')     
        
        
backend = Aer.get_backend('qasm_simulator')
job = execute(qc, backend, shots=99)
result = job.result()
count =result.get_counts()
print(count)

