# https://github.com/stoicswe/CSCI395A-QuantumComputing/blob/7bd82873d4477ca0b4daab96964ebc44b9036671/2d%20Data%20Walk/QWalk2D.py
import matplotlib.pyplot as plt
#%matplotlib inline
import numpy as np
from scipy import linalg as la

#import quantum computing fuctions
from qiskit import QuantumProgram
from qiskit.tools.visualization import plot_histogram, plot_state

quantum = QuantumProgram()
qubit = qp.create_quantum_register("qubit", 4)
classic = qp.create_classical_register("classic", 4)
qc = qp.create_circuit("quantumWalk", [qubit], [classic])

steps = int(argv[1])