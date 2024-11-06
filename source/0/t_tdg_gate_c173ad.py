# https://github.com/anukriti0009/useful_quantum_circuits_with_results/blob/1e26d1dc92f08df3d77da63009cb90a492521fd8/t_tdg_gate.py
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 17:55:36 2022

@author: anukr
"""
import numpy as np
from qiskit import QuantumCircuit, execute, Aer
from qiskit.visualization import plot_histogram
# Use Aer's qasm_simulator
simulator = Aer.get_backend('qasm_simulator')
# Create a Quantum Circuit with 1 Qubit
circuit = QuantumCircuit(1, 1)
# Add a s and sdg gate on Qubit 0
circuit.t(0)   # Apply T-gate to qubit 0
circuit.tdg(0) # Apply Tdg-gate to qubit 0
# Map the quantum measurement to the classical register
circuit.measure([0], [0])
# Execute the circuit on the qasm simulator
job = execute(circuit, simulator, shots=1000)
# Grab results from the job
result = job.result()
# Returns counts
counts = result.get_counts(circuit)
print("\nTotal count for 0 and 1 are:",counts)
# Draw the circuit
print(circuit.draw(output='text'))













