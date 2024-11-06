# https://github.com/stoicswe/CSCI395A-QuantumComputing/blob/7bd82873d4477ca0b4daab96964ebc44b9036671/Quantum%20Interface%20Classifier/quantumInterfaceClassificer.py
import matplotlib.pyplot as plt
#%matplotlib inline
import numpy as np
from scipy import linalg as la

#import quantum computing fuctions
from qiskit import QuantumProgram
from qiskit.tools.visualization import plot_histogram, plot_state

Q_Program = QuantumProgram()

#initialize the registers
qubit = Q_Program.create_quantum_register("qubit", 4)
classic = Q_Program.create_classical_register("classic", 4)
quantumCircuit = Q_Program.create_circuit("quantum_interface_classifier", [qubit], [classic])

quantumCircuit.h(qubit[0])
quantumCircuit.h(qubit[1])

quantumCircuit.cu3(4.304,0,0, qubit[0], qubit[2])#check this

quantumCircuit.x(qubit[0])
quantumCircuit.ccx(qubit[0], qubit[1], qubit[2])
quantumCircuit.x(qubit[1])
quantumCircuit.ccx(qubit[0], qubit[1], qubit[2])

quantumCircuit.u3(-0.6625,0,0, qubit[2])

quantumCircuit.ccx(qubit[0], qubit[1], qubit[2])

quantumCircuit.u3(-0.6625,0,0, qubit[2])

quantumCircuit.cx(qubit[2], qubit[3])
quantumCircuit.cx(qubit[3], qubit[2])
quantumCircuit.cx(qubit[2], qubit[3])
quantumCircuit.cx(qubit[1], qubit[2])

quantumCircuit.h(qubit[0])
quantumCircuit.measure(qubit[0], classic[0])
quantumCircuit.measure(qubit[1], classic[1])

result = Q_Program.execute(["quantum_interface_classifier"], backend='local_qasm_simulator', shots=1024)

#return the result to console: bits and bit states that were counted
print(result)
print(result.get_data("quantum_interface_classifier"))

#plot the counts, determine probability of each state
plot_histogram(result.get_counts('quantum_interface_classifier'))