# https://github.com/MoizAhmedd/quantumtesting/blob/e0643e26628ef1cdc6d0c7309264bddd38004905/gates.py
# quantum_phase.py
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer

# Define the Quantum and Classical Registers
q = QuantumRegister(1)
c = ClassicalRegister(1)

# Build the circuit
circuits = []
pre = QuantumCircuit(q,c)
pre.h(q) #Adds hadamard gate

middle = QuantumCircuit(q,c) #new circuit
meas_x = QuantumCircuit(q,c) #new circuit

meas_x.h(q) #Add hadamard gate to meas_x
meas_x.measure(q,c) #measures meas_x
exp_vector = range(0,8) #performs 8 experiments

for exp_index in exp_vector: # loops 8 times
    circuits.append(pre+middle+meas_x)
    middle.t(q)




# Execute the circuit
shots = 1024
job = execute(circuits,backend = Aer.get_backend('qasm_simulator'),shots = shots,seed = 8)
result = job.result()

# Print the result
for exp_index in exp_vector:
    data = result.get_counts(circuits[exp_index])
    try:
        p0 = data['0']/shots
    except KeyError:
        p0 = 0
    try:
        p1 = data['1']/shots
    except KeyError:
        p0 = 0
    try:
        p1 = data['1']/shots
    except KeyError:
        p1 = 0
    print('exp {}:[{},{}] X length = {}'.format(exp_index,p0,p1,p0-p1))



