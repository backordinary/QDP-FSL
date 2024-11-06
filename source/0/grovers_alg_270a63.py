# https://github.com/cgrant093/Coding-with-Qiskit/blob/9100efdcf837fa2085e5b379e00af0c81ce781f1/Grovers_alg.py
# find an specific element of a list

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit.providers.aer import QasmSimulator
from qiskit.visualization import plot_histogram

# Use Aer's qasm_simulator
simulator = QasmSimulator()

# Create a oracle operator
oracle = QuantumCircuit(2, name='oracle')
oracle.cz(0,1)      #flips sign of winning state, (specific to |11> being the winning state) 
oracle.to_gate()    #makes oracle its own gate

# create reflection operator
reflection = QuantumCircuit(2, name='reflection')
# take our superposition state back to \ell-0 state
reflection.h([0,1]) 
# apply negative phase only to 00 state
reflection.z([0,1]) 
reflection.cz(0,1) 
# transform back to superpos state
reflection.h([0,1]) 
reflection.to_gate() #turns refelction into a gate

# create circuit that flips winning answer: |11>
grover_circ = QuantumCircuit(2,2)
# apply H gate to all qubits
grover_circ.h([0,1]) #prepares superposition state 
grover_circ.append(oracle,[0,1]) # add on oracle
grover_circ.append(reflection,[0,1]) # add on reflection
grover_circ.measure([0,1],[0,1]) # measure

# compile the circuit down to low-level QASM instructions
# supported by the backend (not needed for simple circuits)
compiled_circuit = transpile(grover_circ, simulator)

# Execute the circuit on the qasm simulator
job = simulator.run(compiled_circuit, shots=1)
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