# https://github.com/Julio-Medina/Seminario/blob/0d6d6e7fc0127e5e0ee843a7c522bb339df3b6e1/Qiskit/Grover.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 22:51:55 2022

@author: julio
"""

import numpy as np
from qiskit import IBMQ, QuantumCircuit, Aer, execute
from qiskit.quantum_info import Operator
from qiskit.providers.ibmq import least_busy
from qiskit.visualization import plot_histogram
provider=IBMQ.load_account()

def phase_oracle(n, indices_to_mark, name='Oracle'):
    qc=QuantumCircuit(n, name=name)
    oracle_matrix=np.identity(2**n)
    for index_to_mark in indices_to_mark:
        oracle_matrix[index_to_mark, index_to_mark]=-1
    qc.unitary(Operator(oracle_matrix), range(n))
    return qc
    
def diffuser(n):
    qc=QuantumCircuit(n, name="Diff - V")
    qc.h(range(n))
    qc.append(phase_oracle(n,[0]),range(n))
    qc.h(range(n))
    return qc

def Grover(n, marked):
    qc=QuantumCircuit(n,n)
    r=int(np.round(np.pi/(4*np.arcsin(np.sqrt(len(marked)/2**n)))))
    print(f'{n} qubits, basis state {marked} marked, {r} rounds')
    qc.h(range(n))
    for _ in range(r):
        qc.append(phase_oracle(n,marked), range(n))
        qc.append(diffuser(n), range(n))
    qc.measure(range(n), range(n))
    return qc

n=5
x= np.random.randint(2**n)
marked=[x]
qc=Grover(n, marked)

qc.draw(output='mpl')

backend= Aer.get_backend('qasm_simulator')
result=execute(qc, backend, shots=10000).result()
counts=result.get_counts(qc)
print(counts)
print(np.pi/(4*np.arcsin(np.sqrt(len(marked)/2**n)))-1/2)
plot_histogram(counts)

