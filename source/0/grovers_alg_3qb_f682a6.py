# https://github.com/cgrant093/Coding-with-Qiskit/blob/9100efdcf837fa2085e5b379e00af0c81ce781f1/Grovers_alg_3qb.py
# finds 2 specific elements of a list for 3 qubits (101 and 110)

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit.providers.aer import QasmSimulator
from qiskit.visualization import plot_histogram

# Apply an H-gate to n# of qubits
def initialize_s(QuantumCircuit, qubits):
    for q in qubits:
        QuantumCircuit.h(q)
    return QuantumCircuit

# Create a oracle operator
oracle = QuantumCircuit(3, name='  oracle')
oracle.cz(1,2)      #flips sign of winning state, (specific to |110> being the winning state) 
oracle.cz(0,2)      #flips sign of winning state, (specific to |101> being the winning state) 
oracle.to_gate()    #makes oracle its own gate

# Create reflection/diffusion operator for n qubits
def diffuser(nqubits):
    reflection = QuantumCircuit(nqubits, name='  U_s')
    
    #apply transformation |s> -> |00..0> (H-gates)
    for qubit in range(nqubits):
        reflection.h(qubit)
        
    #apply transformation |00..0> -> |11..1> (X-gates)
    for qubit in range(nqubits):
        reflection.x(qubit)
        
    #multi-controlled-Z gate
    reflection.h(nqubits - 1)
    reflection.mct(list(range(nqubits - 1)), nqubits - 1) #multi-controlled-toffoli
    reflection.h(nqubits - 1)
    
    #apply transformation |11..1> -> |00..0> (X-gates)
    for qubit in range(nqubits):
        reflection.x(qubit)
        
    #apply transformation |00..0> -> |s> (H-gates)
    for qubit in range(nqubits):
        reflection.h(qubit)
        
    #return as a gate
    return reflection.to_gate()
#why is 2 qb use z gate then cont-z for reflection/diffusion
    #but 3+ qb use cont-z surrounded by x-gates?
    #does the general (above) result in the 2 qb case?



# create circuit that flips winning answer: |11>
grover_circ = QuantumCircuit(3,3)
grover_circ = initialize_s(grover_circ, [0,1,2]) 
grover_circ.append(oracle, [0,1,2]) # add on oracle
grover_circ.append(diffuser(3), [0,1,2]) # add on diffuser
grover_circ.measure([0,1,2], [0,1,2]) # measure

# Use Aer's qasm_simulator
simulator = QasmSimulator()

# compile the circuit down to low-level QASM instructions
# supported by the backend (not needed for simple circuits)
compiled_circuit = transpile(grover_circ, simulator)

# Execute the circuit on the qasm simulator
job = simulator.run(compiled_circuit, shots=100)
# Grab results from the job
result = job.result()

# Returns counts
counts = result.get_counts(grover_circ)
print(counts)

# Draw the circuit (with matplotlib)
grover_circ.draw(output='mpl')
plt.show()

# Plot a histogram
#plot_histogram(counts)
#plt.show()