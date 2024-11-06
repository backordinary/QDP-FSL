# https://github.com/JacobViertel/Biped/blob/02ad195a4f7469e27f418cb2b127d6387a1f30a7/shortest%20path/oracle_small.py
import numpy as np
import math

# importing Qiskit
import qiskit
from qiskit import IBMQ, Aer
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, execute, BasicAer

# import basic plot tools
from qiskit.visualization import plot_histogram

def L12(circ,a,b,c):
    circ.x(b)
    circ.ccx(a,b,c)
    circ.x(b)
    circ.x(a)
    circ.ccx(a,b,c)
    circ.x(a)

def L22(circ,a,b,c):
    circ.x(a)
    circ.x(b)
    circ.ccx(a,b,c)
    circ.x(b)
    circ.x(a)
    circ.ccx(a,b,c)

qn = 4
cn = 4
creg = QuantumRegister(qn)
ccreg = ClassicalRegister(qn)
qreg = QuantumRegister(cn)
cqreg = ClassicalRegister(cn)

found_path = False
solution = 0
while (found_path == False):
    circ = QuantumCircuit(creg, qreg, ccreg, cqreg)
    for qubit in range(qn):
        circ.h(qubit)
        
    L12(circ,0,1,qn)#node A
    L12(circ,2,3,qn+1)#node D
    L22(circ,0,2,qn+2)#node B
    L22(circ,1,3,qn+3)#node C
    circ.barrier()
    circ.measure(qreg,cqreg)
    circ.measure(creg,ccreg)
    
    # Execute and see results
    
    emulator = Aer.get_backend('qasm_simulator')
    found_result = False 
    job = execute(circ, emulator, shots=1)
    answer = job.result().get_counts()
    for key, value in answer.items():
        if key[:4] == "1111":
            # print("String ", key)
            # print("Solution ", key[:4])
            print("Path ", key[4:10])
            solution +=1
            if solution > 10:
                found_path = True