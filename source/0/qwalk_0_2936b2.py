# https://github.com/stoicswe/CSCI395A-QuantumComputing/blob/7bd82873d4477ca0b4daab96964ebc44b9036671/quantumWalk_version3/qwalk_0.py
import matplotlib.pyplot as plt
#%matplotlib inline
import numpy as np
from scipy import linalg as la

#import quantum computing fuctions
from qiskit import QuantumProgram
from qiskit.tools.visualization import plot_histogram, plot_state

quantum = QuantumProgram()

qubit = quantum.create_quantum_register("qubit", 7)
classic = quantum.create_classical_register("classic", 4)
qcircuit = quantum.create_circuit("quantumWalk", [qubit], [classic])

qcircuit.h(qubit[0])
qcircuit.cx(qubit[0],qubit[1])
qcircuit.x(qubit[0])
qcircuit.h(qubit[0])
qcircuit.ccx(qubit[0],qubit[1],qubit[2])
qcircuit.cx(qubit[0],qubit[1])
qcircuit.x(qubit[0])
qcircuit.h(qubit[0])
qcircuit.ccx(qubit[0],qubit[1],qubit[6])
qcircuit.ccx(qubit[6],qubit[2],qubit[3])
qcircuit.ccx(qubit[0],qubit[1],qubit[6])
qcircuit.ccx(qubit[0],qubit[1],qubit[2])
qcircuit.cx(qubit[0],qubit[1])
qcircuit.x(qubit[0])
qcircuit.h(qubit[0])
qcircuit.ccx(qubit[0],qubit[1],qubit[6])
qcircuit.ccx(qubit[2],qubit[3],qubit[5])
qcircuit.ccx(qubit[5],qubit[6],qubit[4])
qcircuit.ccx(qubit[2],qubit[3],qubit[5])
qcircuit.ccx(qubit[0],qubit[1],qubit[6])
qcircuit.ccx(qubit[0],qubit[1],qubit[6])
qcircuit.ccx(qubit[6],qubit[2],qubit[3])
qcircuit.ccx(qubit[0],qubit[1],qubit[6])
qcircuit.ccx(qubit[0],qubit[1],qubit[2])
qcircuit.cx(qubit[0],qubit[1])
qcircuit.x(qubit[0])
qcircuit.h(qubit[0])
qcircuit.ccx(qubit[0],qubit[1],qubit[6])
qcircuit.ccx(qubit[2],qubit[3],qubit[5])
qcircuit.ccx(qubit[5],qubit[6],qubit[4])
qcircuit.ccx(qubit[2],qubit[3],qubit[5])
qcircuit.ccx(qubit[0],qubit[1],qubit[6])
qcircuit.ccx(qubit[0],qubit[1],qubit[6])
qcircuit.ccx(qubit[6],qubit[2],qubit[3])
qcircuit.ccx(qubit[0],qubit[1],qubit[6])
qcircuit.ccx(qubit[0],qubit[1],qubit[2])
qcircuit.cx(qubit[0],qubit[1])
qcircuit.x(qubit[0])

qcircuit.measure(qubit[1], classic[0])
qcircuit.measure(qubit[2], classic[1])
qcircuit.measure(qubit[3], classic[2])
qcircuit.measure(qubit[4], classic[3])

#run the complete circuit and gather the result
result = quantum.execute(["quantumWalk"], backend='local_qasm_simulator', shots=1024)

#return the result to console: bits and bit states that were counted
print(quantum.get_qasm("quantumWalk"))
print(result)
print(result.get_data("quantumWalk"))

#plot the counts, determine probability of each state
plot_histogram(result.get_counts('quantumWalk'))