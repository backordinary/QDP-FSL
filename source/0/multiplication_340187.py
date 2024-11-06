# https://github.com/shikhar58/Task1-Multiplier/blob/00e92dc32f5b3f2ddad12f062de34550dfd62800/Multiplication.py
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 11:16:10 2022
@author: shikhar
"""
from qiskit import QuantumCircuit, Aer, QuantumRegister, ClassicalRegister
import qiskit
import numpy as np
from qiskit.circuit.library import QFT

#function for designing the basic quantum circuit before the multiplication operation
def preprocess(first,second):
    #converting number into binary
    a_bit='{0:{fill}3b}'.format(first,fill='0')
    b_bit='{0:{fill}3b}'.format(second,fill='0')
    l1=len(a_bit)
    l2=len(b_bit)
    
    
    r_a=QuantumRegister(l1,'a')
    r_b=QuantumRegister(l2,'b')
    cr=ClassicalRegister(l1, 'c')
    qc=QuantumCircuit(r_a,r_b,cr)
    
    #assigning alues to qubits
    for i in range(l1):
        if a_bit[i]=="1":
            qc.x(r_a[l1-(i+1)])
        if b_bit[i]=="1":
            qc.x(r_b[l2-(i+1)])
    return  qc, r_a, r_b, cr, l1, l2
   
    
#function fo adding two numbers
def addition(inp1,inp2):
    #obtaining the preliminary circuit design
    qc, r_a, r_b, cr, l1, l2 = preprocess(inp1,inp2)
    
    #function for rotations of qubits
    def controlledrotations(qc,r_a,r_b, n):
        for i in range(0,n+1):
            #print(i,n+1)
            qc.cu1(np.pi/2**i,r_b[n-i],r_a[n])

    #applying fourier transformation
    qc.append(QFT(l1,do_swaps=False),[(l1-1)-i for i in range(l1)])
    
    #applying rotation of qubits
    for i in range(0,l2):
        controlledrotations(qc,r_a,r_b,(l2-1)-i)
    
    #inverse fourier transform
    qc.append(QFT(l1,do_swaps=False).inverse(),[(l1-1)-i for i in range(l1)])
    
    #collapsing qbits
    qc.measure(r_a,cr)
    
    #simulating quantum compting
    backend=Aer.get_backend("qasm_simulator")
    job=qiskit.execute(qc,backend,shots=1000)
    result=job.result()
    counts=result.get_counts(qc)
    final=max(counts, key=counts.get)
    return (int(final,2))


#function for multiplication which uses iterative addition
def multiply(inp1,inp2):
    csum = 0
    for i in range(inp2):
        csum=addition(inp1,csum)
    return csum

if __name__ == "__main__":
    answer=multiply(2,2)    
    print ("The answer is", answer)    

