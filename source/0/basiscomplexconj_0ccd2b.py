# https://github.com/Athaagra/temp/blob/6574fa10d71362e8fc95f18caad37840b8b13756/BasisComplexConj.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 31 22:27:42 2021

@author: kel
"""

import numpy as np

a=2+1j
b=3+4j
x=np.array([a, b])

def norm(vector):
    def coeffLength(coeff):
        print('This is coeff {}'.format(coeff))
        print('This is conj coeff {}'.format(np.conj(coeff)))
        return np.real(coeff*np.conj(coeff))
    def totalLength():
        sum=0
        for i in vector:
            print(i)
            print(coeffLength(i))
            sum +=coeffLength(i)
            print('This is the sum {}'.format(sum))
        return sum**(0.5)
    totalSum=np.array(totalLength())
    return vector/totalSum
norm(x)

a=1+1j
b=1-1j
x=np.array([a, b])
def inner(A,B):
    return A*np.matrix(B).getH()

inner(x, x)

def outer(A,B):
    return np.outer(A,B)

outer(x,x)

#Linearly indep orthogonal
matrix = np.array(
    [
    [0, 1, 5, 20],
    [3, 0, 4, 16],
    [0, 1, 9, 36],
    [1, 7, 0, 0],
    ])

vectorA=np.array([1, 1j])
vectorB=np.array([1j, 1])

def isLinIndep(M):
    print(M.shape[1])
    print(np.linalg.matrix_rank(M))
    return(M.shape[1]==np.linalg.matrix_rank(M))

isLinIndep(matrix)

def isOrtho(A, B):
    return(inner(A, B) == 0)

isOrtho(vectorA,vectorB)

M=np.array([
    [1/2**(0.5), 1/2**(0.5)],
    [1/2**(0.5), -1/2**(0.5)],
    ])

def isUnitary(M):
    print('this is the shape M {}'.format(np.shape(M)[0]))
    print('this is the H* M {}'.format(np.matrix(M).getH()*M))
    return np.allclose(np.eye(np.shape(M)[0]), np.matrix(M).getH()*M)

isUnitary(M)

def tensorproduct(A, B):
    return np.kron(A,B)

tensorproduct(M,M)

from qiskit import *
#%matplotlib inline
from math import pi,sqrt
from qiskit.visualization import plot_state_city,plot_bloch_multivector,plot_bloch_multivector

circ = QuantumCircuit(2)

circ.h(0)
circ.x(1)


def getHisto(n, circ):
    qasm_sim=Aer.get_backed('qasm_simulator')
    shots = n
    qobj = assemble(circ, shots=shots)
    results = qasm_sim.run(qobj).result()
    counts = results.get_counts()
    return counts
circ.draw('mpl')
backend = Aer.get_backend('statevector_simulator')
result = execute(circ, backend).result()
output = result.get_statevector(circ, decimals=3)
#plot_histogram(getHisto(1,circ), figsize=(15, 8), bar_labels=False)
plot_state_city(output)

circ2 = QuantumCircuit(6)
circ2.h(1)
circ2.z(1)
circ2.x(2)
circ2.y(3)
circ2.z(4)
circ2.h(5)
circ2.rz(pi/4, 5)
circ2.h(5)

circ2.draw('mpl')
qobj = assemble(circ2)
state=backend.run(qobj).result().get_statevector()
plot_bloch_multivector(state)   

#Random Number Generator from 1 to 64
circ3 =QuantumCircuit(6,6)
circ3.h(0)
circ3.measure(0, 0)
circ3.h(1)
circ3.measure(1, 1)
circ3.h(2)
circ3.measure(2, 2)
circ3.h(3)
circ3.measure(3, 3)
circ3.h(4)
circ3.measure(4, 4)
circ3.h(5)
circ3.measure(5, 5)

circ3.draw('mpl')

measures=[]
outcome = execute(circ3, backend).result()
hist = outcome.get_counts()
for i in hist.keys():
    measures.append(int(i, 2))
outcome = execute(circ3, backend).result()
hist = outcome.get_counts()
for i in hist.keys():
    measures.append(int(i, 2))
outcome = execute(circ3, backend).result()
hist = outcome.get_counts()
for i in hist.keys():
    measures.append(int(i, 2))
outcome = execute(circ3, backend).result()
hist = outcome.get_counts()
for i in hist.keys():
    measures.append(int(i, 2))
outcome = execute(circ3, backend).result()
hist = outcome.get_counts()
for i in hist.keys():
    measures.append(int(i, 2))
outcome = execute(circ3, backend).result()
hist = outcome.get_counts()
for i in hist.keys():
    measures.append(int(i, 2))
outcome = execute(circ3, backend).result()
hist = outcome.get_counts()
for i in hist.keys():
    measures.append(int(i, 2))
outcome = execute(circ3, backend).result()
hist = outcome.get_counts()
for i in hist.keys():
    measures.append(int(i, 2))
outcome = execute(circ3, backend).result()
hist = outcome.get_counts()
for i in hist.keys():
    measures.append(int(i, 2))
outcome = execute(circ3, backend).result()
hist = outcome.get_counts()
for i in hist.keys():
    measures.append(int(i, 2))

matrix=np.array(
    [
     [47, 17, -14],
     [-41, 4, 0],
     [28, 58, 43],
     ])

def get_eigenvals(M):
    print('the eigenvals of {}'.format(np.linalg.eigvals(M)))
    return np.linalg.eigvals(M)

get_eigenvals(matrix)


def get_eigenvectors(M):
    v, v = np.linalg.eig(M)
    print(v,v)
    return v


vectors = get_eigenvectors(matrix)

vectors[:,2].reshape(3, 1)

V = np.array(
    [ 
         [1/2**(0.5), 1/2**(0.5), 1/2**(0.5), 1/2**(0.5)],
    
    ])
axes=[]

axes.append(np.array([
        [1, 0, 0, 0],
    ]))
axes.append(np.array([
        [0, 1, 0, 0],
    ]))
axes.append(np.array([
        [0, 0, 1, 0],
    ]))
axes.append(np.array([
        [0, 0, 0, 1],
    ]))


def proj(V, axis):
    return np.dot(V, np.dot(outer(axis, axis), np.transpose(V)))

proj(V, axes[0])

length=0
for i in axes:
    length +=proj(V, i )
length = sqrt(length)
print(length)


#Bell state
circB = QuantumCircuit(2, 2)
circB.h(0)
circB.cx(0, 1)
circB.measure(0, 0)
circB.measure(1, 1)

circB.draw('mpl')

measures = []
outcome = execute(circB, backend).result()
hist = outcome.get_counts()
for 0 in hist.keys():
    measures.append(0)
outcome = execute(circB, backend).result()
hist = outcome.get_counts()
for 1 in hist.keys():
    measures.append(1)
outcome = execute(circB, backend).result()
hist = outcome.get_counts()
for 2 in hist.keys():
    measures.append(2)
outcome = execute(circB, backend).result()
hist = outcome.get_counts()
for 3 in hist.keys():
    measures.append(3)
outcome = execute(circB, backend).result()
hist = outcome.get_counts()
for 4 in hist.keys():
    measures.append(4)
outcome = execute(circB, backend).result()
hist = outcome.get_counts()
for 5 in hist.keys():
    measures.append(5)
outcome = execute(circB, backend).result()
hist = outcome.get_counts()
for 6 in hist.keys():
    measures.append(6)
outcome = execute(circB, backend).result()
hist = outcome.get_counts()
for 7 in hist.keys():
    measures.append(7)
outcome = execute(circB, backend).result()
hist = outcome.get_counts()
for 8 in hist.keys():
    measures.append(8)
outcome = execute(circB, backend).result()
hist = outcome.get_counts()
for 9 in hist.keys():
    measures.append(9)

print(measures)

b=[0,1,0,0,0,0,1,1,1,1,1,1,0,1,1,0,1,0,1,0]
#density matrix
qr = QuantumRegister(2)
qc = QuantumCircuit(qr)
qc.initialize([0,1,0,0], qr) #,0,0,0,0], qr)#,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], qr)#,0,0,1,1], qr)#,1,1,1,1,0,1,1,0,1,0,1,0], qr)
qc.z(0)
qc.z(1)
#qc.z(2)
#qc.z(3)
#qc.z(4)
#qc.z(5)
#qc.z(6)
#qc.z(7)
#qc.z(8)
#qc.z(9)
#qc.z(10)
#qc.z(11)
#qc.z(12)
#qc.z(13)
#qc.z(14)
#qc.z(15)
#qc.z(16)
#qc.z(17)
#qc.z(18)
#qc.z(19)
qc.draw('mpl')


from qiskit.quantum_info import DensityMatrix
#Bell state
state = execute(qc, backend).result().get_statevector()
state
DensityMatrix(state)


def getDensity(quantumCircuit):
    return np.matrix(DensityMatrix(execute(quantumCircuit, backend).result().get_statevector()).data)

getDensity(qc)
#b=[0,1,0,0,0,0,1,1,1,1,1,1,0,1,1,0,1,0,1,0]
#circ0=QuantumCircuit(20)
#for i in range(0,len(b)):
    #circ0.h(i)
#    circ0.z(i)
    #circ0.measure(i,i)
#circ0.draw('mpl')
#backend = Aer.get_backend('aer_simulator')
#state = execute(circ0, backend).result().get_statevector()
#print(state)
#DensityMatrix(state)
#getDensity(circ0)
#measures = []
#for i in range(0, 10):
#    outcome = execute(circ0, backend).result()
#    hist = outcome.get_counts()
#    for i in hist.keys():
#        measures.append(i)

#partial trace component of the bloch vector
quantun_info.partial_trace(state, [0])

def getPtrace(quantumCircuit, n):
    return quantum_info.partial_trace(execute(quantumCircuit, backend).result().get_statevector(), [n])

getPTrace(qc,0)

def getRx(pd):
    sigmaX=np.matrix([[0, 1],[1, 0]])
    return np.trace(pd.data*sigmaX)

def getRy(pd):
    sigmaY=np.matrix([[0, 1j],[1j, 0]])
    return np.trace(pd.data*sigmaY)

getRx(getPTrace(qc, 1))

def getRz(pd):
    sigmaZ=np.matrix([[1, 0],[0, -1]])
    return np.trace(pd.data*sigmaZ)

getRz(getPTrace(qc, 1))

getRy(getPTrace(qc, 0))

#toffoli, swap, shift

qreq = QuantumRegister(10)
qclass=ClassicalRegister(10)
backend = Aer.get_backend('statevector_simulator')
quantumC = QuantumCircuit(qreq,qclass)
initializedState = [0 for i in range(0, 2**10)]
initializedState[102]=1
quantumC.initialize(initializedState, qreq)
plot_bloch_multivector(execute(quantumC, backend).result().get_statevector())

quantumC.ccx(0, 1, 2)
quantumC.swap(3, 4)
quantumC.h(6)
quantumC.cp(pi/3, 5, 6)

plot_bloch_multivector(execute(quantumC, backend).result().get_statevector())
quantumC.draw('mpl')

#Fredkin gate
qreq = QuantumRegister(3)
qclass = ClassicalRegister(3)
quantumC = QuantumCircuit(qreq, qclass)
initializedState = [0 for i in range(0, 2**3)]
initializedState[5]=1
quantumC.initialize(initializedState, qreq)
plot_bloch_multivector(execute(quantumC, backend).result().get_statevector())
quantumC.cswap(0, 1, 2)
quantumC.cx(1, 2)
quantumC.ccx(0, 2, 1)
quantumC.cx(1, 2)
quantumC.draw('mpl')
plot_bloch_multivector(execute(quantumC, backend).result().get_statevector())

