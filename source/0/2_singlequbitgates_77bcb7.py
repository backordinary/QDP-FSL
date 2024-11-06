# https://github.com/VashuKochar/QiskitProjects/blob/c16ba4e4e0832f4ba5678e3ee7899163b27fd7a1/Labs/2_singleQubitGates.py
"""
## Single Qubit Gates

### Source
https://learn.qiskit.org/course/ch-labs/lab-2-single-qubit-gates

### Problem statement
Create quantum circuits to apply various single qubit gates on different states and understand the change in state and phase of the qubit.

### References
QuantumCircuit: https://qiskit.org/documentation/stubs/qiskit.circuit.QuantumCircuit.html
compose: https://medium.com/arnaldo-gunzi-quantum/how-to-use-the-compose-function-in-qiskit-5983bf4fffcf
AER Simulator: https://qiskit.org/documentation/tutorials/simulators/1_aer_provider.html
barrier: https://quantumcomputing.stackexchange.com/questions/8369/what-is-a-barrier-in-qiskit-circuits
"""


import numpy as np

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
    # We encode '0' as the qubit state |0⟩, and '1' as |1⟩
    # Since the qubit is initially |0⟩, we don't need to do anything for an input of '0'
    # For an input of '1', we do an x to rotate the |0⟩ to |1⟩
    if inp=='1':
        qc.x(range(nq))
    elif inp=='+':
        qc.h(range(nq))
    elif inp=='-':
        qc.x(range(nq))
        qc.h(range(nq))
    elif inp=='i':
        qc.h(range(nq))
        qc.s(range(nq))
    elif inp=='-i':
        qc.x(range(nq))
        qc.h(range(nq))
        qc.s(range(nq))
        
    return qc

def printStates(qc):
    backend = Aer.get_backend('statevector_simulator')
    out1 = execute(qc,backend).result().get_statevector()
    print(out1)
    
    return qc, out1

# Tests

# Part 1: Effect of Single-Qubit Gates on state |0>
# qc1 = initCircuit(4, '0')
# Part 2: Effect of Single-Qubit Gates on state |1>
qc1 = initCircuit(4, '1')
# Part 3: Effect of Single-Qubit Gates on state |+>
# qc1 = initCircuit(4, '+')
# Part 4: Effect of Single-Qubit Gates on state |->
# qc1 = initCircuit(4, '-')
# Part 5: Effect of Single-Qubit Gates on state |i>
# qc1 = initCircuit(4, 'i')
# Part 6: Effect of Single-Qubit Gates on state |-i>
# qc1 = initCircuit(4, '-i')

qc1.barrier()
# perform gate operations on individual qubits
qc1.x(0)
qc1.y(1)
qc1.z(2)
qc1.s(3)

qc1.barrier()

# Draw circuit
print(qc1)

# Plot blochshere
out1 = printStates(qc1)