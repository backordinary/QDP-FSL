# https://github.com/sqrta/Qsketch/blob/5f3057dc5bd8b1a466644a5173c967c73edb642f/try/scipy_try.py
from scipy.optimize import minimize
import numpy as np
from math import cos,sin,pi,sqrt
from math import asin as arcsin
from qiskit.quantum_info import Operator

from qiskit import QuantumRegister, ClassicalRegister
from qiskit import QuantumCircuit, execute
from qiskit import Aer

backend = Aer.get_backend('unitary_simulator')

def Ry(theta):
    return np.array([
        [cos(theta/2), -sin(theta/2)],
        [sin(theta/2), cos(theta/2)]
    ])

def Rx(theta):
    return np.array([
        [cos(theta/2), -sin(theta/2)],
        [sin(theta/2), cos(theta/2)]
    ])

gatelist = [Ry]

initial = np.array([1,0,0,0])
final = np.array([sqrt(2)/sqrt(3), 0, sqrt(1)/sqrt(3), 0])

def f(x):
    
    gate = np.kron(Ry(x[0]), I)
    gate2 = np.kron(Rx(x[1]), I)
    norm = np.linalg.norm(gate2.dot(gate.dot(initial))- final)
    return norm

def g(x):
    circ= QuantumCircuit(2)
    circ.ry(x[0], 0)
    unitary = Operator(circ).data
    norm = np.linalg.norm(unitary.dot(initial)- final)
    return norm

theta = 2*arcsin(1/sqrt(3))
I = np.array([
    [1,0],
    [0,1]
])
# gate = np.kron(Ry(theta), I)
# print(gate.dot(initial))
# circ= QuantumCircuit(2)
# circ.ry(theta, 0)

# unitary = Operator(circ)
# print(unitary.data)
mymin = minimize(g,[0], method="BFGS")
print(mymin)
print(theta)

