# https://github.com/Ecaterina-Hrib/Quantum-Computing/blob/9daff5b15c025ac2e7de57c2ba7b636188a706b1/tema%202/ex1.py
import numpy as np
from qiskit import (QuantumCircuit,QuantumRegister,ClassicalRegister,execute,Aer)
from qiskit.visualization import *
import math
import matplotlib.pyplot as plotter
a = 9 
N = 55
variabile= [0]*100
def f(a,x,N):
   a=a**x
   a=a%N
   return a
interval = range(1, 100)
for i in interval:
    variabile[i]=0
    variabile[i]=f(a,i,N)
print(variabile)
plotter.plot(variabile)
plotter.show()

  

