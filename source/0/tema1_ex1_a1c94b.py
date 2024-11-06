# https://github.com/Ecaterina-Hrib/Quantum-Computing/blob/12265e032bf51ea7bd24277b7fee86d18f75ebbd/tema1-ex1.py
import numpy as np
from qiskit import (QuantumCircuit,QuantumRegister,ClassicalRegister,execute,Aer)
from qiskit.visualization import *
#nr de qubiti
n=2
qr=n
#nr de biti
cr=n
backend=Aer.get_backend('statevector_simulator')
#simulator=Aer.get_backend('qasm_simulator')
circuit=QuantumCircuit(qr,cr)
#poarta X pt qubitul 1 pentru a schimba valoarea din 0 in 1
circuit.x(1)
#controlled NOT-targhetul pe primul si controlul pe al doilea
circuit.cx(0,1)
circuit.cx(1,0)
circuit.cx(0,1)
#create quantum program for execution

for i in range(n):
 circuit.measure(i,i)
print(circuit.draw())
#execut circuitul cu quasm
job=execute(circuit,backend)
#iau rezultatele
result=job.result()
counts=result.get_counts(circuit)
print("rezultatele sunt",counts)
plot_histogram(counts).savefig('histogram.png')
