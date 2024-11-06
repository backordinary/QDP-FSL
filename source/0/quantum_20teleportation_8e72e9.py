# https://github.com/neilgupte75/CS_810_Quantum-Computing_Neil/blob/67697b0288014c72ab65277a04881c48c7351a19/quantum%20teleportation.py
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 19:44:30 2020

@author: Neil Gupte
""" 
from qiskit import(QuantumCircuit,  execute,  Aer)
from qiskit.visualization import plot_histogram
# Use Aer's qasm_simulator
simulator = Aer.get_backend('qasm_simulator')
circuit = QuantumCircuit(3, 3)
circuit.draw(output="mpl")

circuit.x(0)
 
circuit.draw()
circuit.barrier()

circuit.draw()
circuit.h(1)
circuit.cx(1,2)
circuit.draw()
circuit.cx(0,1)
circuit.h(0)
circuit.draw()
circuit.measure([0,1], [0,1])
circuit.cx(1,2)
circuit.cx(0,2)
circuit.draw()
circuit.measure(2,2)
job1 = execute(circuit, simulator, shots=1000)
# Grab results from the job
result1 = job1.result()
# Returns counts
counts1 = result1.get_counts(circuit)
print("\nTotal count for 10 and 01 are:",counts1)
plot_histogram(result1.get_counts(circuit))