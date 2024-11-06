# https://github.com/Ecaterina-Hrib/Quantum-Computing/blob/12265e032bf51ea7bd24277b7fee86d18f75ebbd/tema1-ex2.py
import numpy as np
from qiskit import (QuantumCircuit,QuantumRegister,ClassicalRegister,execute,Aer)
from qiskit.visualization import *
#Controlled-Z matrice cu CNOT si Hadamard si CNOT din C-Z cu Hadamard
#nr de qubiti
n=2
qr=n
#nr de biti
cr=n
#backend=Aer.get_backend('statevector_simulator')
simulator=Aer.get_backend('qasm_simulator')
dem1=QuantumCircuit(qr,cr)
circuit=QuantumCircuit(qr,cr)
dem2=QuantumCircuit(qr,cr)
circuit2=QuantumCircuit(qr,cr)
dem1.x(1)
dem1.cz(0,1)

dem2.x(0)
dem2.cx(0,1)

circuit.x(1)
# HXH=Z
#Hadamard
circuit.h(1)
#CNOT
circuit.cx(0,1)
#H
circuit.h(1)


circuit2.x(0)
# HZH=X
circuit2.h(1)
#C-Z
circuit2.cz(0,1)
#H
circuit2.h(1)

for i in range(n):
 dem1.measure(i,i)
 dem2.measure(i,i)
 circuit.measure(i,i)
 circuit2.measure(i,i)
print(dem1.draw())
print(dem2.draw())
print(circuit.draw())
print(circuit2.draw())
job=execute(dem1,simulator,shots=1000)
#iau rezultatele
result=job.result()
counts=result.get_counts(dem1)
print("rezultatele la CZ sunt",counts)

job=execute(dem2,simulator,shots=1000)
#iau rezultatele
result=job.result()
counts=result.get_counts(dem2)
print("rezultatele la CX sunt",counts)

job=execute(circuit,simulator,shots=1000)
#iau rezultatele
result=job.result()
counts=result.get_counts(circuit)
print("rezultatele la HXH sunt",counts)

job=execute(circuit2,simulator,shots=1000)
#iau rezultatele
result=job.result()
counts=result.get_counts(circuit2)
print("rezultatele la HZH sunt",counts)
