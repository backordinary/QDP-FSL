# https://github.com/MightyGoldenOctopus/QCOMP/blob/5f06fec4bab01be17eade0a7d5455c8ee0cd87bb/Workshop/WP1/correction.py
import numpy as np 
from enum import Enum
import math
from qiskit import QuantumCircuit, execute, Aer


nm = np.array([0,1,1])
print(nm)
print(np.array_equal(np.transpose(nm),nm))
#Enumration for each type of gate our quantum computer handle
class TypeOfQuantumGate(Enum):
    NOT = 1
    HADAMARD = 2 
    CNOT = 3

#The class quantum gate
#TypeOfGate is the type of the gate (a value of TypeOfQuantumGate)
#fQbit is the position in the circuit of the first input Qbit of the gate
#sQbit is the position in the circuit of the second input Qbit of the gate ( if the gate has two input)
class QuantumGate:
    def __init__(self, typeOfGate: TypeOfQuantumGate, fQbit: int, sQbit: int =0 ):
        self.typeOfGate = typeOfGate
        self.fQbit = fQbit
        self.sQbit = sQbit


#The main function, you need to program it without using qiskit
#nbQbits is the number of Qbits of the circuit
#QuantumGates is the list of quantum gates of the circuits
#This function output the state vector of the circuit after executing all the gate
def quantumComputer(nbQbits: int, quantumGates: list):
    tab = np.zeros([2**nbQbits])#, dtype = np.dtype(np.complex128))
    tab[0] = 1
    a = 1 / math.sqrt(2)
    for i in quantumGates:
        if(i.typeOfGate == TypeOfQuantumGate.NOT):
            tab = np.matmul(computeMatrix(np.array([[0,1],[1,0]]), nbQbits, i.fQbit) , np.transpose(tab))
        elif(i.typeOfGate == TypeOfQuantumGate.HADAMARD):
            tab = np.matmul(computeMatrix(np.array([[a,a],[a,-a]]), nbQbits, i.fQbit) , np.transpose(tab))
        elif(i.typeOfGate == TypeOfQuantumGate.CNOT):
            tab = np.matmul(computeCNOT( nbQbits, i.fQbit, i.sQbit) , np.transpose(tab))
    return tab

arr = np.array([1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j])
comparison = np.array_equal(arr, quantumComputer(3,[]))
assert(comparison)

def kroneckerProduct(m1,m2):
    return np.kron(m1,m2)

m1 = [1,2]
m2 = [[4,5],[6,7]]
assert(np.array_equal(kroneckerProduct(m1,m2) , [[ 4,  5,  8, 10],[ 6,  7, 12, 14]]))


def computeMatrix(baseMatrix, nbQbit, fQbit):
     res = np.identity(1)
     for j in range(nbQbit):
         i = nbQbit - 1 - j
         if(i == fQbit):
             res = np.kron(res, baseMatrix)
         else:
             res = np.kron(res, np.array([[1,0],[0,1]]))
     return res

m = [[0,1],[1,0]]
arr = [[0., 1., 0., 0.],
 [1., 0., 0., 0.],
 [0., 0., 0., 1.],
 [0., 0., 1., 0.]]
assert(np.array_equal(computeMatrix(m,2,0), arr))

arr =quantumComputer(2,[QuantumGate(TypeOfQuantumGate.NOT,0)])
assert(np.array_equal(arr,[0., 1., 0., 0.]))


arr =quantumComputer(1,[QuantumGate(TypeOfQuantumGate.HADAMARD,0)])
assert(np.isclose(arr,[1/math.sqrt(2), 1/math.sqrt(2) ]).all())   

#test multiple gate
arr =quantumComputer(1,[QuantumGate(TypeOfQuantumGate.NOT,0),QuantumGate(TypeOfQuantumGate.HADAMARD,0)])
assert(np.isclose(arr,[1/math.sqrt(2), -1/math.sqrt(2) ]).all())   




def check(i, c):
    i = i >>c 
    return i %2

assert(check(15,2)) 
assert(not check(16,2)) 

def oppo(i, t):
  a = 1 << t 
  return a ^ i
assert(oppo(16,3) == 24) 
assert(oppo(15,3) == 7)


def computeCNOT(nbQubits, fQubit, sQubit):
    size = 2**nbQubits
    res = np.zeros([size, size])
    for i in range(size):
        if check(i,fQubit):
            res[i][oppo(i,sQubit)] = 1
        else :
            res[i][i] = 1
    return res
arr = [[1., 0., 0., 0.],
 [0., 0., 0., 1.],
 [0., 0., 1., 0.],
 [0., 1., 0., 0.]]
assert(np.array_equal(computeCNOT(2,0,1) , arr))

arr =quantumComputer(2,[QuantumGate(TypeOfQuantumGate.NOT,1),QuantumGate(TypeOfQuantumGate.CNOT,1,0)])
assert(np.array_equal(arr , [0,0,0,1]))

def computeProbability(tab):
    res = np.zeros([tab.size])
    sum = 0
    for i in tab:
        sum += abs(i)**2
    for i in range(0,tab.size):
        res[i] = (abs(tab[i])**2)/sum
    return res

arr =quantumComputer(1,[QuantumGate(TypeOfQuantumGate.NOT,0),
                        QuantumGate(TypeOfQuantumGate.HADAMARD,0)])
assert(np.isclose(computeProbability(arr),[0.5,0.5]).all())
