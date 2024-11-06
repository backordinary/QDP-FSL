# https://github.com/kishan-shanthakumar/Projects/blob/82db9fa2d710772fb9f8849e086f9da44efc1e82/Python/Quantum%20Computing/adder.py
from qiskit import QuantumRegister
from qiskit import ClassicalRegister
from qiskit import QuantumCircuit
from qiskit import execute
from qiskit import *
first=input("Enter a number")
second=input("Enter a number")
l1=len(first)
l2=len(second)
n=max(l1,l2)
creg=ClassicalRegister(num_bits)
qreg=QuantumRegister(num_bits)
circ=QuantumCircuit(qreg,creg)
a=QuantumRegister(n)
b=QuantumRegister(n+1)
c=QuantumRegister(n)
c1=ClassicalRegister(n+1)
qc=QuantumCircuit(a,b,c,c1)
circ.x(qreg[1])
for i in range(l1):
    if first[i]=="1":
        qc.x(a[l1-(i+1)])
for i in range(l2):
    if second[i]=="1":
        qc.x(b[l2-(i+1)])
for i in range(n-1):
    qc.ccx(a[i],b[i],c[i+1])
    qc.cx(a[i],b[i])
    qc.ccx(c[i],b[i],c[i+1])
qc.ccx(a[n-1],b[n-1],c[n])
qc.cx(a[n-1],b[n-1])
qc.ccx(c[n-1],b[n-1],c[n])
#bit.ly/205Mgm9
