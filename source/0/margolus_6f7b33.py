# https://github.com/ntanetani/FRQI/blob/2c85fd4f78ecd0d0c14ff38349c35e583323511a/src/margolus.py
from qiskit import IBMQ, QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import execute, QuantumRegister
from qiskit.qasm import pi
from qiskit.tools.visualization import plot_histogram, circuit_drawer
from qiskit import execute, Aer, BasicAer
import numpy as np
import random
import matplotlib as mpl

backends = Aer.backends()
print("Aer backends:",backends)

q = QuantumRegister(3)
c = ClassicalRegister(3)
qc = QuantumCircuit(q,c)


for i in range(1,len(q)):
        qc.h(q[i])

'''
for i in range(len(q)):
        qc.measure(q[i],c[i])
'''

backend_sim = Aer.get_backend('qasm_simulator')

print(qc.depth())
numOfShots = 1024000
#result = execute(qc, backend_sim, shots=numOfShots).result()
circuit_drawer(qc, output='mpl', plot_barriers=False).savefig('circ.png', dpi=400)
#plot_histogram(result.get_counts(qc))
