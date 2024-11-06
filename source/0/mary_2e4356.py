# https://github.com/ntanetani/FRQI/blob/2c85fd4f78ecd0d0c14ff38349c35e583323511a/src/mary.py
from qiskit import IBMQ, QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import execute, QuantumRegister
from qiskit.qasm import pi
from qiskit.tools.visualization import plot_histogram, circuit_drawer
from qiskit import execute, Aer, BasicAer
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

q = QuantumRegister(3)
c = ClassicalRegister(3)
qc = QuantumCircuit(q,c)

def margolus(circ, t, c0, c1):
    circ.ry(np.pi/4,t)
    circ.cx(c0, t)
    circ.ry(np.pi/4,t)
    circ.cx(c1, t)
    circ.ry(-np.pi/4,t)
    circ.cx(c0, t)
    circ.ry(-np.pi/4,t)

def mary00(circ, angle, t, c0, c1):
    circ.ry(angle/2,t)
    circ.cx(c1, t)
    circ.ry(angle/2,t)
    circ.cx(c0, t)
    circ.ry(-angle/2,t)
    circ.cx(c1, t)
    circ.ry(-angle/2,t)
    circ.cx(c0, t)
    circ.x(c0)

def mary01(circ, angle, t, c0, c1):
    circ.ry(angle/2,t)
    circ.cx(c0, t)
    circ.ry(-angle/2,t)
    circ.cx(c1, t)
    circ.ry(-angle/2,t)
    circ.cx(c0, t)
    circ.ry(angle/2,t)
    circ.cx(c1, t)

def mary10(circ, angle, t, c0, c1):
    circ.ry(angle/2,t)
    circ.cx(c0, t)
    circ.ry(angle/2,t)
    circ.cx(c1, t)
    circ.ry(-angle/2,t)
    circ.cx(c0, t)
    circ.ry(-angle/2,t)
    circ.cx(c1, t)

def mary11(circ, angle, t, c0, c1):
    circ.ry(angle/2,t)
    circ.cx(c0, t)
    circ.ry(-angle/2,t)
    circ.cx(c1, t)
    circ.ry(angle/2,t)
    circ.cx(c0, t)
    circ.ry(-angle/2,t)
    circ.cx(c1, t)

for i in range(1,len(q)):
    qc.h(q[i])

mary00(qc, np.pi/2, q[0], q[1], q[2])
mary01(qc, np.pi/3, q[0], q[1], q[2])
mary10(qc, np.pi/4, q[0], q[1], q[2])
mary11(qc, np.pi/8, q[0], q[1], q[2])
qc.tdg(0)
for i in range(len(q)):
    qc.measure(q[i],c[i])

backend_sim = Aer.get_backend('qasm_simulator')
#print(qc.depth())
numOfShots = 4096
result = execute(qc, backend_sim, shots=numOfShots).result()
#plot_histogram(result.get_counts(qc))

print(result.get_counts(qc))