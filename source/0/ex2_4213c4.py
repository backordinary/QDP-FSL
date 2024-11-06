# https://github.com/Ecaterina-Hrib/Quantum-Computing/blob/12265e032bf51ea7bd24277b7fee86d18f75ebbd/ex2.py
import numpy as np
from qiskit import (QuantumCircuit,QuantumRegister,ClassicalRegister,execute,Aer)
from qiskit.visualization import *
import math
n=4
qr=n
cr=n
circuit=QuantumCircuit(qr,cr)
circuit.x(3)
circuit.y(3)
circuit.x(3)

circuit.h(0)
circuit.h(1)
circuit.h(2)
#j1=0,j2=1,j3=0

circuit.cry(0.04*math.pi,0,3)
circuit.cry(0.04*math.pi,1,3)
circuit.cry(0.04*math.pi,1,3)
circuit.cry(0.04*math.pi,2,3)
circuit.cry(0.04*math.pi,2,3)
circuit.cry(0.04*math.pi,2,3)
circuit.cry(0.04*math.pi,2,3)

circuit.h(0)
circuit.h(1)
circuit.h(2)

circuit.crx(-math.pi/4,2,0)
circuit.crx(-math.pi/2,1,0)
circuit.crx(-math.pi/2,2,1)

for i in range(n-1):
 circuit.measure(i,i)
print(circuit.draw())
simulator=Aer.get_backend('qasm_simulator')
job=execute(circuit,simulator)
result=job.result()
counts=result.get_counts()
print(counts)