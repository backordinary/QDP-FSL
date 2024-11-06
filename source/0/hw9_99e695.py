# https://github.com/neilgupte75/CS_810_Quantum-Computing_Neil/blob/67697b0288014c72ab65277a04881c48c7351a19/hw9.py
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 14:16:04 2020

@author: Neil Gupte
"""

from qiskit import(QuantumCircuit,  execute,  Aer)
from qiskit.visualization import plot_histogram
# Use Aer's qasm_simulator
simulator = Aer.get_backend('qasm_simulator')
circuit1 = QuantumCircuit(3, 3)
circuit1.x(0)
circuit1.draw()
circuit1.barrier()


circuit1.h(1)
circuit1.draw()

circuit1.cx(1,2)
circuit1.draw()


circuit1.cx(0,1)
circuit1.draw()


circuit1.h(0)
circuit1.draw()

circuit1.barrier()


circuit1.measure(0,0)

circuit1.measure(1,1)

circuit1.barrier()
circuit1.draw()

circuit1.cx(1,2)
circuit1.draw()

circuit1.cz(0,2)
circuit1.draw()
job1 = execute(circuit1, simulator, shots=1000)
# Grab results from the job
result1 = job1.result()
# Returns counts
counts1 = result1.get_counts(circuit1)
print("\nTotal counts",counts1)
plot_histogram(result1.get_counts(circuit1))




