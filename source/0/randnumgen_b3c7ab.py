# https://github.com/p5ynk3d-p5yk1x/IBM-Qiskit-Random-Number-Generator/blob/4478d2c628ebf0f720179d15c241274338895b3f/randnumgen.py
import qiskit as q
from qiskit import QuantumCircuit, execute, Aer
import numpy as np
import cmath
def randbool(): 
    qc = QuantumCircuit(1,1)
    qc.h(0)
    qc.measure(0,0)
    be = Aer.get_backend('statevector_simulator')
    jobs = execute(qc,be, shots = 1)
    results = jobs.result()
    qcm = results.get_statevector()
    return(qcm[0].real)
def inttobin(n):
    binary = []
    temp = n
    while int(temp) != 0 :
        binary.append(int(temp%2))
        temp = temp/2
    return(binary)

def bintoint(x):
    sum = 0
    for i in range(len(x)):
        sum = sum + (pow(2,i)*x[i])
    return(sum)
    
def randbingen(n):
    bruh = []
    for i in range(n):
        bruh.append(randbool())
    return(bruh)
    
def randrange(ll,ul):
    l=inttobin(ll)
    u=inttobin(ul)
    if len(l)==len(u):
        n = randbingen(len(l))
        num = bintoint(n)
        if num > ul or num < ll:
            return(randrange(ll,ul))
        elif ll<num<ul:
            return(num)
    else :
        ex = 0
        for i in range(len(u)-len(l)):
            ex = ex + randbool()
        n = randbingen(len(l)+int(ex))
        num = bintoint(n)
        print(n,num)
        if num > ul or num < ll:
            return(randrange(ll,ul))
        elif ll<num<ul:
            return(num)
