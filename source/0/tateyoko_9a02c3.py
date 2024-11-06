# https://github.com/ntanetani/FRQI/blob/2c85fd4f78ecd0d0c14ff38349c35e583323511a/src/tateyoko.py
from qiskit import IBMQ, QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import execute, QuantumRegister
from qiskit.qasm import pi
from qiskit.tools.visualization import plot_histogram, circuit_drawer
from qiskit import execute, Aer, BasicAer
import numpy as np
import matplotlib.pyplot as plt


tateshima = [[255,0],[255,0]]
plt.imshow(tateshima, cmap='gray')
plt.show()

def ccfrqi(circ, controls, angle, t, c0, c1):
    
    clist = []
    for i in controls:
            clist.append(int(i))

    if clist[0] == 0:
            circ.x(c0)
    if clist[1] == 0:
            circ.x(c1)

    circ.cu3(angle/2, 0, 0, c1, t)
    circ.cx(c1, c0)
    circ.cu3(-angle/2, 0, 0, c0, t)
    circ.cx(c1, c0)
    circ.cu3(angle/2, 0, 0, c0, t)

    if clist[0] == 0:
            circ.x(c0)
    if clist[1] == 0:
            circ.x(c1)
