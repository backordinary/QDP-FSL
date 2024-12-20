# https://github.com/mcditoos/QCHACK2021/blob/fcc011a6b23d2081a971e3ffc036df61e1185d75/circuit.py
import matplotlib.pyplot as plt
import numpy as np
from qiskit import *




def ansatz(ansatzList,theta=3.1415):
    q = QuantumRegister(2)
    c = ClassicalRegister(2)
    circuit = QuantumCircuit(q, c)
    for gate in ansatzList:
        if gate[0]=='H':
            p=int(gate[1])
            circuit.h(q[p])
        if gate[0]=='C':
            p0=int(gate[1])
            p1=int(gate[2])
            circuit.cx(q[p0], q[p1])
        """   
        if gate[0]=='R':
            p=int(gate[1])
            circuit.rx(theta, q[p])
             """
        if gate[0]=='Y':
            p=int(gate[1])
            circuit.y(q[p])
        if gate[0]=='Z':
            p=int(gate[1])
            circuit.z(q[p])
        if gate[0]=='X':
            p=int(gate[1])
            circuit.x(q[p])
    circuit.measure(q,c)
    return circuit






def get_expectation(ansatzList, theta=3.1415):

    circuit = ansatz(ansatzList, theta)
    
    shots = 10000 
    backend = BasicAer.get_backend('qasm_simulator')
    job = execute(circuit, backend, shots=shots)
    result = job.result()
    counts = result.get_counts()
        
    return counts

def comparison(ansatzList,theta=3.1415):
    estimate=get_expectation(ansatzList,theta)
    total=sum(estimate.values())
    if '11' in estimate.keys():
        return True
    else:
        return False