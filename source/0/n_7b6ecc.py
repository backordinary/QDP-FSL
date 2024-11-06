# https://github.com/Yathish27/Machine-Learning-Algorithms/blob/188fdcb71cad541dd8ef5a7c2ed01445b246c43d/n.py
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 15:25:26 2019

@author: YATHISH N V
"""
import qiskit
from qiskit import QuantumRegister
from qiskit import ClassicalRegister
from qiskit import QuantumCircuit

first = input("Enter a binary number:")
second = input("Enter another binary number:")
l1=len(first)
l2=len(second)
n=max(l1,l2)
creg=ClassicalRegister(n)
qreg=QuantumRegister(n)
Circ=QuantumCircuit(qreg,creg)
a=QuantumRegister(n)
b=QuantumRegister(n+1)
c=QuantumRegister(n)
cl=ClassicalRegister(n+1)
qc=QuantumCircuit(a,b,c,cl)
"""circ.x(qreg[1])"""
for i in range(l1):
    if first[i] == "1":
         qc.x(a[l1-(i+1)])
for i in range(l2):
    if second[i] == "1":
         qc.x(b[l2-(i+1)])
for i in range(n-1):
    qc.ccx(a[i],b[i],c[i+1])
    qc.cx(a[i],b[i])
    qc.ccx(c[i],b[i],c[i+1])
qc.ccx(a[n-1],b[n-1],b[n])
qc.cx(a[n-1],b[n-1])
qc.ccx(c[n-1],b[n-1],b[n])
qc.cx(c[n-1],b[n-1])
for i in range(n-1):
    qc.ccx(c[(n-2)-i],b[(n-2)-i],c[(n-2)-i])
    qc.cx(a[(n-2)-i],b[(n-2)-i])
    qc.ccx(a[(n-2)-i],b[(n-2)-i],c[(n-2)-i])
    qc.cx(c[(n-2)-i],b[(n-2)-i])
    qc.cx(a[n-1],b[n-1])
