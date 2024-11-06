# https://github.com/cgrant093/Coding-with-Qiskit/blob/e7e37688d33bbb432ee7503f368eb636afe66cfc/Bernstein_Vazirani_alg.py
# will guess a secret 6-digit number in ONE try

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit.providers.aer import QasmSimulator
from qiskit.visualization import plot_histogram

# set secret number 
secretNumber = '10100101'

# Create a Quantum Circuit acting on the q register
circuit = QuantumCircuit(len(secretNumber)+1, len(secretNumber))

# apply H gate to all but last qubit
circuit.h(range(len(secretNumber)))
# last gate gets X gate then H gate
circuit.x(len(secretNumber))
circuit.h(len(secretNumber))
circuit.barrier()

# build box with secret number
#   any bit that has a '1' gets a controlled X gate 
#   applied with last qubit (remember bits read backwards)

for ii, yesno in enumerate(reversed(secretNumber)):
    if yesno == '1':
        circuit.cx(ii,len(secretNumber))
        
circuit.barrier()

# apply H gate to first 6 qubits
circuit.h(range(len(secretNumber)))
circuit.barrier()

# now to measure
circuit.measure(range(len(secretNumber)),range(len(secretNumber)))

# Use Aer's qasm_simulator
simulator = QasmSimulator()

# compile the circuit down to low-level QASM instructions
# supported by the backend (not needed for simple circuits)
compiled_circuit = transpile(circuit, simulator)

# Execute the circuit on the qasm simulator
job = simulator.run(compiled_circuit, shots=1)

# Grab results from the job
result = job.result()

# Returns counts
counts = result.get_counts(circuit)
print(counts)

# Draw the circuit (with matplotlib)
#circuit.draw(output='mpl')
#plt.show()

# Plot a histogram
#plot_histogram(counts)
#plt.show()