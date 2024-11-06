# https://github.com/SPL-LSU/Codes/blob/a28db11b399e6175134e55b973997b67fa44b0df/RoyWIP/qiskitplayground.py
#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# %load qiskitplayground.py
"""
Created on Mon Feb 10 18:49:57 2020
A notebook wherein I learn to play with python.
@author: Margarite L. LaBorde
"""

import qutip as qt
from qiskit import *
import numpy as np

#takes as input a basis qutip state, output the state creation circuit
def translate(qutipvector):
    vector=qutipvector;
    if vector.type == 'ket':
        n=log2(vector.dims[0][0])
        new=QuantumCircuit(n)
        temp=vector.full()
        for index in range(n):
            if temp[index][0] >0:
                new.x(index)
    else:
        print("y'all this only works with kets please don't try too hard")
    
    return new

def tensor_fix(gate):
    result = gate.full()
    result = qt.Qobj(result)
    return result

#takes a number of qubits, a vector of which should be initialized to the 1 state (zero-indexed), 
#and whether the first is in zero or one
def rabbit(qubits,choice,start):
    basic_0ket=qt.Qobj([[1],[0]])
    basic_1ket=qt.Qobj([[0],[1]])
    if start ==1:
        temp=basic_1ket
    else:
        temp=basic_0ket
    r=1
    while r < qubits:
        if r in choice:
            temp=qt.tensor(temp,basic_1ket)
            temp=tensor_fix(temp)
        else:
            temp=qt.tensor(temp,basic_0ket)
            temp=tensor_fix(temp)
        r+=1
    return temp

def state_creation_circuit(index,qubits):
    if index == 1:
        state_create=QuantumCircuit(qubits)
    elif index ==2:
        state_create=QuantumCircuit(qubits)
        state_create.x(0)
    elif index == 3:
        state_create=QuantumCircuit(qubits)
        state_create.x(qubits)
        state_create.x(0)
    elif index == 4:
        state_create=QuantumCircuit(qubits)
        for i in range(qubits):
            state_create.x(i)
    return state_create

def hada_all_qubits(circ, qubits):
    circuit=circ
    for i in range(qubits):
        circuit.h(i)
    return circuit

def main():
    """
    test=qt.Qobj(np.transpose([1,0,1,0]));
    circ=translate(test)
    #circ.cx(1,3)
    m=circ.width()
    if m >0:
        print("wowie!")
        print(circ.draw())
    else:
        print("man...")
    """
    new=state_creation_circuit(4,4)
    new2=hada_all_qubits(new,4)
    #new.x(1)
    print(new2.draw())
    return 0

main()


# In[ ]:




