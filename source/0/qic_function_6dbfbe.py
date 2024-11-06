# https://github.com/stoicswe/CSCI395A-QuantumComputing/blob/7bd82873d4477ca0b4daab96964ebc44b9036671/Quantum%20Machine%20Learning/Quantum%20Version%201%20-%204Qubits/QIC_function.py
import matplotlib.pyplot as plt
#%matplotlib inline
import numpy as np
from scipy import linalg as la

#import quantum computing fuctions
from qiskit import QuantumProgram
from qiskit.tools.visualization import plot_histogram, plot_state

def makeQubits(quantumProgram, qubitRegisters): #return qubit
    print("Added [" + str(qubitRegisters) + "] qubits to the quantum register")
    return quantumProgram.create_quantum_register("qubit", qubitRegisters)

def makeClassical(quantumProgram, classicalRegisters): #return classical
    print("Added [" + str(classicalRegisters) + "] bits to the classical register")
    return quantumProgram.create_classical_register("classic", classicalRegisters)

def makeCircuit(quantumProgram, qubit, classic): #return complete circuit
    print("Generating base quantum circuit...")
    return quantumProgram.create_circuit("quantum_interface_classifier", [qubit], [classic])
    

def initializeCircuit(quantumCircuit, startValue, qubit):
    print("Initializing quantum circuit...")
    quantumCircuit.h(qubit[0])
    quantumCircuit.h(qubit[1])
    quantumCircuit.cu3(startValue,0,0, qubit[0], qubit[2])
    quantumCircuit.x(qubit[0])
    return quantumCircuit

#function add data point
def addDataPoint(quantumCircuit, theta, qubit):
    quantumCircuit.ccx(qubit[0], qubit[1], qubit[2])
    quantumCircuit.cu3(theta,0,0, qubit[0], qubit[2])
    quantumCircuit.x(qubit[1])
    quantumCircuit.ccx(qubit[0], qubit[1], qubit[2])
    quantumCircuit.cu3(theta,0,0, qubit[0], qubit[2])
    quantumCircuit.x(qubit[1])
    print("Controlled U function added to circuit with value: " + str(theta))
    return quantumCircuit

#function finalize circuit
def finalizeCircuit(quantumCircuit, qubit, classic):
    print("Finalizing quantum circuit...")
    quantumCircuit.ccx(qubit[0], qubit[1], qubit[2])
    quantumCircuit.u3(-0.6625,0,0, qubit[2])
    quantumCircuit.ccx(qubit[0], qubit[1], qubit[2])
    quantumCircuit.u3(-0.6625,0,0, qubit[2])
    quantumCircuit.cx(qubit[2], qubit[3])
    quantumCircuit.cx(qubit[3], qubit[2])
    quantumCircuit.cx(qubit[2], qubit[3])
    quantumCircuit.cx(qubit[1], qubit[2])
    quantumCircuit.h(qubit[0])
    print("Adding quantum measuring functions...")
    quantumCircuit.measure(qubit[0], classic[0])
    quantumCircuit.measure(qubit[1], classic[1])
    return quantumCircuit

def getQASM(quantumProgram):
    return quantumProgram.get_qasm("quantum_interface_classifier")

#function run
def runCircuit(quantumProgram, shotCount):
    print("Executing quantum program with " + str(shotCount) + " shots...")
    result = quantumProgram.execute(["quantum_interface_classifier"], backend='local_qasm_simulator', shots=shotCount)

    #return the result to console: bits and bit states that were counted
    print(result)
    print(result.get_data("quantum_interface_classifier"))

    #plot the counts, determine probability of each state
    qcounts = result.get_counts('quantum_interface_classifier')
    try:
        norm0000 = qcounts['0000'] / (qcounts['0000'] + qcounts['0010'])
        norm0010 = qcounts['0010'] / (qcounts['0000'] + qcounts['0010'])
        print("0000 Normalized: " + str(norm0000))
        print("0010 Normalized: " + str(norm0010))
    except:
        print("Only counted [0000]: " + str(qcounts['0000']))
    plot_histogram(result.get_counts('quantum_interface_classifier'))