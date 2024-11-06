# https://github.com/JacobViertel/Biped/blob/02ad195a4f7469e27f418cb2b127d6387a1f30a7/shortest%20path/oracle.py
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

def L13(circ,a,b,c,d,e):
    circ.x(b)
    circ.x(c)
    circ.ccx(a,b,d)
    circ.ccx(c,d,e)
    circ.ccx(a,b,d)
    circ.x(b)
    circ.x(c)
    circ.x(a)
    circ.x(c)
    circ.ccx(a,b,d)
    circ.ccx(c,d,e)
    circ.ccx(a,b,d)
    circ.x(a)
    circ.x(c)
    circ.x(a)
    circ.x(b)
    circ.ccx(a,b,d)
    circ.ccx(c,d,e)
    circ.ccx(a,b,d)
    circ.x(a)
    circ.x(b)

def L22(circ,a,b,c):
    circ.x(a)
    circ.x(b)
    circ.ccx(a,b,c)
    circ.x(b)
    circ.x(a)
    circ.ccx(a,b,c)

def L23(cric,a,b,c,d,e):
    circ.x(a)
    circ.x(b)
    circ.x(c)
    circ.ccx(a,b,d)
    circ.ccx(c,d,e)
    circ.ccx(a,b,d)
    circ.x(c)
    circ.x(b)
    circ.x(a)
    
    circ.x(c)
    circ.ccx(a,b,d)
    circ.ccx(c,d,e)
    circ.ccx(a,b,d)
    circ.x(c)

    circ.x(b)
    circ.ccx(a,b,d)
    circ.ccx(c,d,e)
    circ.ccx(a,b,d)
    circ.x(b)

    circ.x(a)
    circ.ccx(a,b,d)
    circ.ccx(c,d,e)
    circ.ccx(a,b,d)
    circ.x(a)
    
qn = 7
cn = 6
an = 2
creg = QuantumRegister(qn)
ccreg = ClassicalRegister(qn)
qreg = QuantumRegister(cn)
cqreg = ClassicalRegister(cn)
areg = QuantumRegister(an)
solution = 0
found_path = False
found_solution = []
while (found_path == False):
    circ = QuantumCircuit(creg, qreg, areg,ccreg, cqreg)
    for qubit in range(qn):
        circ.h(qubit)
    L12(circ,0,1,qn)#node A
    L22(circ,0,2,qn+1) #node B
    L22(circ,1,4,qn+2) #node C
    L23(circ,2,3,5,qn+cn,qn+3) #node D
    L22(circ,3,6,qn+4) #node E
    L13(circ,4,5,6,qn+cn+1,qn+5) #node F
    circ.measure(qreg,cqreg)
    circ.measure(creg,ccreg)

# Execute and see results

    emulator = Aer.get_backend('qasm_simulator')
    found_result = False 
    job = execute(circ, emulator, shots=1)
    answer = job.result().get_counts()
    for key, value in answer.items():
        if key[:6] == "111111":
            # print("Solution ", key)
            solution +=1
            if key[6:14] in found_solution:
                pass
                # print("Found already exsisting solution!")
            else:
                found_solution.append(key[6:14])
            if solution > 5:
                found_path = True
print(found_solution)