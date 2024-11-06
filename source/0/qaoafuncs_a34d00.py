# https://github.com/JaimePSantos/Dissertation-Tex-Code/blob/15544a4334f61e670d1eeee9849fd168c468863d/Coding/Qiskit/AllSearch/qaoaFuncs.py
import numpy as np
import matplotlib.pyplot as plt
from qiskit import *
from qiskit.visualization import plot_histogram
from qiskit import IBMQ
from qiskit.tools.monitor import job_monitor
from scipy import linalg

def toket(N,st):
    mat = eye(N)
    state = zeros(N)
    state = mat[:,st]
    return state

def outercalc(N,st1,st2):
    a=toket(N,st1)
    b=toket(N, st2)
    return outer(a,b)

def matrix(marked,marked2,gamma,N,adjunct):
    c = np.zeros((2**N,2**N))
    c[marked][marked] = -1.0*(gamma)
    #c[marked2][marked2] = -1.0*(2*gamma/3)
    #c = c*gamma
    # matprint(c)
    if adjunct==True :
        mat = linalg.expm(-1j*c)
    else:
        mat = linalg.expm(1j*c)
#     print()
#     print()
#    matprint(mat)
#     print()
#     print()
#     print(mat.diagonal())
#     print()
#     print()
    
    return mat

def oracleQAOA(N,mat):
    qreg = QuantumRegister(N)
    qc = QuantumCircuit(qreg)
    qc.diagonal(mat.tolist(),qreg)
    qc = transpile(qc,optimization_level=3)
    return qc

def runQAOA(N,angle,marked,gamma,steps):
    qreg = QuantumRegister(N)
    creg = ClassicalRegister(N)
    qc = QuantumCircuit(qreg,creg)
    marked2= 0

    mat = matrix(marked,marked2,gamma,N,True).diagonal()
    qcaux = oracleQAOA(N,mat)
    qcaux = transpile(qcaux,optimization_level=3)
    mat2 = matrix(marked,marked2,gamma,N,False).diagonal()
    qcaux2 = oracleQAOA(N,mat2)
    qcaux2 = transpile(qcaux2,optimization_level=3)


    qc.h(qreg)
    for t in range(steps):
        qc.append(qcaux,range(N))
        qc.rx(2*angle,qreg)
        qc.append(qcaux2,range(N))
        qc.rx(angle,qreg)
        qc.barrier()
        
    qc.measure(qreg,creg)
    
    return qc

