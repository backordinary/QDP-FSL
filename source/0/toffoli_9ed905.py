# https://github.com/paniash/progs/blob/b7f58efeb5b5c7942af6b8af12611bdbc1a52840/toffoli.py
from qiskit import *
from qiskit.visualization import *
import matplotlib.pyplot as plt
import numpy as np

qc = QuantumCircuit(3)

def hadamard(qc, index):
    qc.rz(np.pi/2, index)
    qc.sx(index)
    qc.rz(np.pi/2, index)

    return qc

def tdagger(qc, index):
    qc.rz(-np.pi/4, index)

    return qc

def tnormal(qc, index):
    qc.rz(np.pi/4, index)

    return qc

# Toffoli gate
circuit = hadamard(circuit, 2)
circuit.cx(1,2)
circuit = tdagger(circuit, 2)
circuit.cx(0,2)
circuit = tnormal(circuit, 2)
circuit.cx(1,2)
circuit = tdagger(circuit, 2)
circuit.cx(0,2)
circuit = tnormal(circuit, 1)
circuit = tnormal(circuit, 2)
circuit.cx(0,1)
circuit = hadamard(circuit, 2)
circuit = tnormal(circuit, 0)
circuit = tdagger(circuit, 1)
circuit.cx(0,1)

qc.draw()
plt.show()
