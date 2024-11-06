# https://github.com/algas/quantum-computing-beginning-guide-book/blob/d9440354bae42516b185a889475b0f3514b172ac/src/quantum_gate/grover.py
import math
import itertools
import sympy
import numpy

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import execute, Aer

def get_result(qc, shots=1):
    simulator = Aer.get_backend('qasm_simulator')
    return execute(qc, simulator, shots=shots).result().get_counts(qc)

def get_unitary(qc):
    simulator = Aer.get_backend('unitary_simulator')
    return execute(qc, simulator).result().get_unitary(qc)

def marking2(qc, k):
    if k == '11':
        qc.cz(0,1)
    elif k == '10':
        qc.x(0)
        qc.cz(0,1)
        qc.x(0)
    elif k == '01':
        qc.x(1)
        qc.cz(0,1)
        qc.x(1)
    else:
        qc.x([0,1])
        qc.cz(0,1)
        qc.x([0,1])

def amp2(qc):
    qc.h([0,1])
    qc.x([0,1])
    qc.cz(0,1)
    qc.x([0,1])
    qc.h([0,1])

def grover2(k):
    q2 = QuantumRegister(2)
    c2 = ClassicalRegister(2)
    qc2 = QuantumCircuit(q2,c2)
    qc2.h([0,1])
    marking2(qc2, k)
    amp2(qc2)
    print(qc2)
    qc2.measure(q2,c2)
    print(get_result(qc2, 100))

def exec_grover2():
    print("Grover's Algorithm 2bit")
    grover2('00')
    grover2('01')
    grover2('10')
    grover2('11')

def ccz(qc, c0, c1, t):
    qc.cx(c1,t)
    qc.tdg(t)
    qc.cx(c0,t)
    qc.t(t)
    qc.cx(c1,t)
    qc.tdg(t)
    qc.cx(c0,t)
    qc.t(c1)
    qc.cx(c0,c1)
    qc.t(c0)
    qc.tdg(c1)
    qc.cx(c0,c1)

def marking3(qc, k):
    if k == '111':
        ccz(qc,0,1,2)
    if k == '110':
        qc.x(0)
        ccz(qc,0,1,2)
        qc.x(0)
    if k == '101':
        qc.x(1)
        ccz(qc,0,1,2)
        qc.x(1)
    if k == '100':
        qc.x([0,1])
        ccz(qc,0,1,2)
        qc.x([0,1])
    if k == '011':
        qc.x(2)
        ccz(qc,0,1,2)
        qc.x(2)
    if k == '010':
        qc.x([0,2])
        ccz(qc,0,1,2)
        qc.x([0,2])
    if k == '001':
        qc.x([1,2])
        ccz(qc,0,1,2)
        qc.x([1,2])
    if k == '000':
        qc.x([0,1,2])
        ccz(qc,0,1,2)
        qc.x([0,1,2])

def amp3(qc):
    qc.h([0,1,2])
    qc.x([0,1,2])
    ccz(qc,0,1,2)
    qc.x([0,1,2])
    qc.h([0,1,2])

def grover3(k):
    q3 = QuantumRegister(3)
    c3 = ClassicalRegister(3)
    qc3 = QuantumCircuit(q3,c3)
    qc3.h([0,1,2])
    marking3(qc3, k)
    amp3(qc3)
    marking3(qc3, k)
    amp3(qc3)
    print(qc3)
    qc3.measure(q3,c3)
    print(get_result(qc3, 100))

def exec_grover3():
    print("Grover's Algorithm 3bit")
    grover3('000')
    grover3('001')
    grover3('010')
    grover3('011')
    grover3('100')
    grover3('101')
    grover3('110')
    grover3('111')

if __name__ == "__main__":
    exec_grover2()
    exec_grover3()
