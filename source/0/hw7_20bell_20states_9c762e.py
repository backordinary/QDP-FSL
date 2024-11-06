# https://github.com/neilgupte75/CS_810_Quantum-Computing_Neil/blob/67697b0288014c72ab65277a04881c48c7351a19/hw7%20bell%20states.py
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 17:14:04 2020

@author: Neil Gupte
"""

import numpy as np 
from qiskit import(QuantumCircuit,  execute,  Aer)
from qiskit.visualization import plot_histogram
# Use Aer's qasm_simulator
simulator = Aer.get_backend('qasm_simulator')
# Create a Quantum Circuit acting on the q register
circuit = QuantumCircuit(2, 2)
# Add a H gate on qubit 0
circuit.h(0)
# Add a CX (CNOT) gate on control qubit 0 and target qubit 1
circuit.cx(0, 1)
# Map the quantum measurement to the classical bits
circuit.measure([0,1], [0,1])
# Execute the circuit on the qasm simulator
job = execute(circuit, simulator, shots=1000)
# Grab results from the job
result = job.result()
# Returns counts
counts = result.get_counts(circuit)
print("\nTotal count for 00 and 11 are:",counts)
plot_histogram(result.get_counts(circuit))
# Draw the circuit
circuit.draw()
##################################################################|01>
circuit1 = QuantumCircuit(2, 2)
circuit1.h(0)
 
circuit1.draw()
circuit1.x(1)

circuit1.cx(0, 1)
circuit1.draw()
circuit1.measure([0,1], [0,1])
job1 = execute(circuit1, simulator, shots=1000)
# Grab results from the job
result1 = job1.result()
# Returns counts
counts1 = result1.get_counts(circuit1)
print("\nTotal count for 10 and 01 are:",counts1)
plot_histogram(result1.get_counts(circuit1))


##################################################################|10>
circuit2 = QuantumCircuit(2, 2)
circuit2.x(0)

circuit2.h(0)
 
circuit2.draw()

circuit2.cx(0, 1)
circuit2.draw()
circuit2.measure([0,1], [0,1])
job2 = execute(circuit2, simulator, shots=1000)
# Grab results from the job
result2 = job2.result()
# Returns counts
counts2 = result2.get_counts(circuit2)
print("\nTotal count for 00 and 11 are:",counts2)
plot_histogram(result2.get_counts(circuit2))

##################################################################|11>
circuit3 = QuantumCircuit(2, 2)
circuit3.x(0)

circuit3.h(0)
 
circuit3.draw()
circuit3.x(1)
circuit3.cx(0, 1)
circuit3.draw()
circuit3.measure([0,1], [0,1])
job3 = execute(circuit3, simulator, shots=1000)
# Grab results from the job
result3 = job3.result()
# Returns counts
counts3 = result3.get_counts(circuit3)
print("\nTotal count for 10 and 01 are:",counts3)
plot_histogram(result3.get_counts(circuit3))


