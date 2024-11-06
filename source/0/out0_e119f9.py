# https://github.com/skushagra/quantum/blob/911161315a44563c97fdd1ee4c2ffacc5030ce78/python_files/out0.py
# For a n-bit input it results a n-bit 0 output. 
# Regardless of the n-bit input it will return a n-bit 0 output.
import qiskit
from qiskit import *
import matplotlib
from qiskit.tools.visualization import plot_histogram
i = input()
circuit = QuantumCircuit(len(i), len(i))
for j in range(0,len(i)):
    circuit.h(j)
circuit.x(0)
circuit.x(len(i)-1)
for j in range(1,len(i)-1,2):
    circuit.cx(j,j+1)
for j in range(0,len(i)):
    circuit.h(j)
# circuit.draw()
backend = Aer.get_backend('qasm_simulator')
result = execute(circuit, backend=backend, shots=1024).result()
counts = result.get_counts()
plot_histogram(counts)
