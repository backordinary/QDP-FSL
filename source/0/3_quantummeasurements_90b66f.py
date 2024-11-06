# https://github.com/VashuKochar/QiskitProjects/blob/c16ba4e4e0832f4ba5678e3ee7899163b27fd7a1/Labs/3_quantumMeasurements.py
"""
## Single Qubit Gates

### Source
https://learn.qiskit.org/course/ch-labs/lab-3-quantum-measurements

### Problem statement
1.  Measuring the state of a qubit
Build the circuits to measure the expectation values of X,Y,Z gate
Estimate the bloch sphere coordinates of the qubit using the qasm simulator


### Math
Express the expectation values of the Pauli operators for an arbitrary qubit state in the computational basis. 
<Z> = <q|Z|q> = <q|0><0|q> - <q|1><1|q> = |<0|q>|^2 - |<1|q>|^2
<X> = <q|X|q> = <q|0><1|q> + <q|1><0|q> =  
<Y> = <q|Y|q> = (-<q|0><1|q> + <q|1><0|q>)i = 


"""


import numpy as np
from numpy import linalg as la

# Importing standard Qiskit libraries
from qiskit import QuantumCircuit, transpile, Aer, IBMQ, execute
from qiskit.visualization import *
from ibm_quantum_widgets import *
from qiskit.providers.aer import QasmSimulator

# Loading your IBM Quantum account(s)
provider = IBMQ.load_account()

def initCircuit(nq, inp):    
    # Initialise circuit
    qc = QuantumCircuit(nq)
    qc.reset(range(nq))
    
    # Encoding the input
    for i in range(len(inp)):
        if inp[i]=='1':
            qc.x(nq-i-1)
        elif inp[i]=='+':
            qc.h(nq-i-1)
        elif inp[i]=='-':
            qc.x(nq-i-1)
            qc.h(nq-i-1)
        elif inp[i]=='i':
            qc.h(nq-i-1)
            qc.s(nq-i-1)
        elif inp[i]=='-i':
            qc.x(nq-i-1)
            qc.h(nq-i-1)
            qc.s(nq-i-1)
        
    return qc

def printStates(qc):
    backend = Aer.get_backend('statevector_simulator')
    out1 = execute(qc,backend).result().get_statevector()
    print(out1)
    
    return qc, out1

# Tests
qc = QuantumCircuit(1)
qc.reset(0)
# qc.x(0)

# z measurement of qubit 0
measure_z = QuantumCircuit(1,1)
measure_z.measure(0,0)

# x measurement of qubit 0
measure_x = QuantumCircuit(1,1)
measure_x.h(0)
measure_x.measure(0,0)

# y measurement of qubit 0
measure_y = QuantumCircuit(1,1)
measure_y.sdg(0)
measure_y.h(0)
measure_y.measure(0,0)

shots = 2**14 # number of samples used for statistics
sim = Aer.get_backend('qasm_simulator')
bloch_vector_measure = []
for measure_circuit in [measure_x, measure_y, measure_z]:
    
    # run the circuit with a the selected measurement and get the number of samples that output each bit value
    counts = execute(qc+measure_circuit, sim, shots=shots).result().get_counts()

    # calculate the probabilities for each bit value
    probs = {}
    for output in ['0','1']:
        if output in counts:
            probs[output] = counts[output]/shots
        else:
            probs[output] = 0
            
    bloch_vector_measure.append( probs['0'] -  probs['1'] )

# normalizing the bloch sphere vector
bloch_vector = bloch_vector_measure/la.norm(bloch_vector_measure)

print('The bloch sphere coordinates are [{0:4.3f}, {1:4.3f}, {2:4.3f}]'
      .format(*bloch_vector))