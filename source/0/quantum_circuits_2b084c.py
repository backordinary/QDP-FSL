# https://github.com/Rohan-s18/Python-Cookbook/blob/dcc3bb99dd02e958dcba8842c94e9361502426e9/Qiskit/Introduction/quantum_circuits.py
"""
Author: Rohan Singh
1/29/23
This Python module contains source code simple Quantum Circuits
"""

#  Imports 
from qiskit import QuantumCircuit, assemble, Aer
from qiskit.visualization import plot_histogram

#  Creating a Quantum Circuit with 8 qubits (8 outputs)
qc_output = QuantumCircuit(8)

#  Adding measurements to each of the qubits
qc_output.measure_all()

#  The qubits are all intialized to 0, so the output will be all zeros
print(qc_output.draw(initial_state=True))



sim = Aer.get_backend('aer_simulator') 
result = sim.run(qc_output).result()
counts = result.get_counts()

#  Printing the counts
print(counts)
plot_histogram(counts)


