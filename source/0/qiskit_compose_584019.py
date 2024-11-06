# https://github.com/asgunzi/qiskit_compose/blob/476e5e910f9fa232fe48ac73745b317c92aa35ed/qiskit_compose.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 01:32:01 2021

Reference
#https://qiskit.org/documentation/stubs/qiskit.circuit.QuantumCircuit.html#qiskit.circuit.QuantumCircuit.compose

@author: Arnaldo Gunzi
"""

from qiskit import QuantumCircuit


n = 4
n_q = n
n_b = n


qc_1 = QuantumCircuit(n_q,n_b)
qc_1.x(0)
qc_1.barrier()
print(qc_1)

qc_2 = QuantumCircuit(n_q,n_b)
qc_2.h(2)
for j in range(n):
    qc_2.measure(j,j)
print(qc_2)

#The + is a deprecated method
#qc = qc_1 + qc_2

#Using compose
qc = qc_1.compose(qc_2)
#qc = qc_1.compose(qc_2, qubits =[0,2,1,3]) #example of changing the wiring of the circuit

print(qc)


