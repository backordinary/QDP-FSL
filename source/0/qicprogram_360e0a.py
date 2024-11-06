# https://github.com/stoicswe/CSCI395A-QuantumComputing/blob/7bd82873d4477ca0b4daab96964ebc44b9036671/Quantum%20Machine%20Learning/Quantum%20Version%201%20-%204Qubits/QICProgram.py
import matplotlib.pyplot as plt
#%matplotlib inline
import numpy as np
import pandas as pd
import math
from scipy import linalg as la

#import quantum computing fuctions
from qiskit import QuantumProgram
#from qiskit.tools.visualization import plot_histogram, plot_state
from DATAImport import getData
from QIC_function import makeQubits, makeClassical, makeCircuit, initializeCircuit, addDataPoint, finalizeCircuit, runCircuit, getQASM

quantum = QuantumProgram()
qubits = makeQubits(quantum, 4)
classical = makeClassical(quantum, 4)
qcircuit = makeCircuit(quantum, qubits, classical)
qcircuit = initializeCircuit(qcircuit, 3.036, qubits)

dataPoints = getData()

#run for loop here to add the points
for x in range(0, len(dataPoints)):
    thetaVar = math.acos(dataPoints[x,0])*2
    print("Adding theta point: " + str(thetaVar))
    qcircuit = addDataPoint(qcircuit, thetaVar, qubits)

print("Generating quantum circuit...")
qcircuit = finalizeCircuit(qcircuit, qubits, classical)
QASM_source = getQASM(quantum)
print(QASM_source)
print("Executing quantum circuit...")
runCircuit(quantum, 1024)