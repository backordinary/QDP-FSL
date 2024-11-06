# https://github.com/parkergarrison/quantum-computing/blob/dfb70ba42dfb9746534048fd1f526e48f7abeaf5/SafeCircuit/test_bits.py
#!/usr/bin/python3
# Reference: This code is based off of the Qiskit tutorial, available at https://hub.mybinder.org/user/qiskit-qiskit-tutorial-f094xh78/notebooks/community/algorithms/grover_algorithm.ipynb (https://mybinder.org/v2/gh/QISKit/qiskit-tutorial/master?filepath=index.ipynb)

import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

# importing Qiskit
from qiskit import BasicAer, IBMQ
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, execute, compile
from qiskit.tools.visualization import plot_histogram

from SafeCircuit import *

def bits2_works_safe(circuit, f_in, f_out, aux, n):
    # see if you can two-color the straight graph of length 4
    # make sure color 0 is vertex 0
    
    # check connection 0-1
    
    sc = SafeCircuit(circuit)
    sc.add_op('cx', f_in[0], aux[0])
    sc.add_op('cx', f_in[1], aux[0])
    
    
    inv2 = SafeCircuit(circuit)
    inv2.add_op('x', f_in[2])
    inv2.add_op('cx', f_in[2], aux[1], dirty=True)
    
    
    print("this is inv2: \n" + "\n".join(str(e) for e in inv2.l))
    
    sc.add_op('cx', f_in[1], aux[1])
    sc.add_cir(inv2)
    
    sc.add_op('ccx', aux[0], aux[1], f_out[0], dirty=True) # !

    sc.write()
    
    print("Executing: \n" + "\n".join(str(e) for e in sc.oplist))
    

def n_controlled_Z(circuit, controls, target, aux2):
    """Implement a Z gate with multiple controls"""
    # edited to allow 5 inputs using 2 ancilla bits
    if (len(controls) > 5):
        raise ValueError('The controlled Z with more than 5 ' +
                         'controls is not implemented')
    elif (len(controls) == 1):
        circuit.h(target)
        circuit.cx(controls[0], target)
        circuit.h(target)
    elif (len(controls) == 2):
        circuit.h(target)
        circuit.ccx(controls[0], controls[1], target)
        circuit.h(target)
    elif (len(controls) == 3):
        circuit.h(target)
        circuit.ccx(controls[0], controls[1], aux2[0])
        circuit.ccx(controls[2], aux2[0], target)
        circuit.ccx(controls[0], controls[1], aux2[0])
        circuit.h(target)
    elif (len(controls) == 4):
        circuit.h(target)
        circuit.ccx(controls[0], controls[1], aux2[0])
        circuit.ccx(controls[2], aux2[0], aux2[1])
        circuit.ccx(controls[0], controls[1], aux2[0])
        
        circuit.ccx(controls[3], aux2[1], target)
        
        circuit.ccx(controls[0], controls[1], aux2[0])
        circuit.ccx(controls[2], aux2[0], aux2[1])
        circuit.ccx(controls[0], controls[1], aux2[0])
        
        circuit.h(target)
    
    elif (len(controls) == 5):
        # we could alternatively do this with more auxiliary qubits or with setting two, and if that at a time first
        circuit.h(target)
        circuit.ccx(controls[0], controls[1], aux2[0]) # >A1 (0,1) 0 is set
        circuit.ccx(controls[2], aux2[0], aux2[1]) # >A2 (0,1,2) answer in 1, can reset 0
        circuit.ccx(controls[0], controls[1], aux2[0]) # <A1 0 is reset
        
        circuit.ccx(controls[3], aux2[1], aux2[0]) # >A3 (0,1,2,3) answer in 0, can reset 1
        
        circuit.ccx(controls[4], aux2[0], target) # !A4 (0,1,2,3,4)
        
           
        circuit.ccx(controls[3], aux2[1], aux2[0]) # <A3
        
        circuit.ccx(controls[0], controls[1], aux2[0]) # >A1 (0,1) 0 is set
        circuit.ccx(controls[2], aux2[0], aux2[1]) # <A2 (0,1,2) answer in 1, can reset 0
        circuit.ccx(controls[0], controls[1], aux2[0]) # <A1 0 is reset   
        
        circuit.h(target)
    
        
# -- end function

def inversion_about_average(circuit, f_in, n, aux2):
    """Apply inversion about the average step of Grover's algorithm."""
    # Hadamards everywhere
    for j in range(n):
        circuit.h(f_in[j])
    # D matrix: flips the sign of the state |000> only
    for j in range(n):
        circuit.x(f_in[j])
    n_controlled_Z(circuit, [f_in[j] for j in range(n-1)], f_in[n-1], aux2)
    for j in range(n):
        circuit.x(f_in[j])
    # Hadamards everywhere again
    for j in range(n):
        circuit.h(f_in[j])
# -- end function

"""
Grover search implemented in Qiskit, from the Qiskit tutorial

This module contains the code necessary to run Grover search on 3
qubits, both with a simulator and with a real quantum computing
device. This code is the companion for the paper
"An introduction to quantum computing, without the physics",
Giacomo Nannicini, https://arxiv.org/abs/1708.03684. 
"""

def input_state(circuit, f_in, f_out, n):
    """(n+1)-qubit input state for Grover search."""
    for j in range(n):
        circuit.h(f_in[j])
    circuit.x(f_out)
    circuit.h(f_out)
# -- end function

# Make a quantum program for the n-bit Grover search.
n = 3

# Exactly-1 3-SAT formula to be satisfied, in conjunctive
# normal form. We represent literals with integers, positive or
# negative, to indicate a Boolean variable or its negation.
exactly_1_3_sat_formula = [[1, 2, -3], [-1, -2, -3], [-1, 2, 3], [1, 2, -3]]

# Define three quantum registers: 'f_in' is the search space (input
# to the function f), 'f_out' is bit used for the output of function
# f, aux are the auxiliary bits used by f to perform its
# computation.
f_in = QuantumRegister(n)
f_out = QuantumRegister(1)
aux = QuantumRegister(9) #len(exactly_1_3_sat_formula) + 1)
aux2 = QuantumRegister(2)

# Define classical register for algorithm result
ans = ClassicalRegister(n)

# Define quantum circuit with above registers
grover = QuantumCircuit()
print(grover.name)
grover.add_register(f_in)
grover.add_register(f_out)
grover.add_register(aux)
grover.add_register(aux2)
grover.add_register(ans)

input_state(grover, f_in, f_out, n)
T = 2
for t in range(T):
    # Apply T full iterations
    #black_box_u_f(grover, f_in, f_out, aux, n, exactly_1_3_sat_formula)
    #example_is010_v0(grover, f_in, f_out, aux, n)
    inversion_about_average(grover, f_in, n, aux2)
    bits2_works_safe(grover, f_in, f_out, aux, n)

# Measure the output register in the computational basis
for j in range(n):
    grover.measure(f_in[j], ans[j])

# Execute circuit
backend = BasicAer.get_backend('qasm_simulator')
job = execute([grover], backend=backend, shots=100)
result = job.result()

# Get counts and plot histogram
counts = result.get_counts(grover)
print(counts)
plot_histogram(counts)
