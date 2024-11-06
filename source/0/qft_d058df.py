# https://github.com/witseie-elen4022/ELEN4022_LAB2_2022_-Chiraira-_-Isaiah-/blob/b1f8f92d5fa472f9afe095f1d7b11e5350c5e8dc/QFT.py
from qiskit import *
import numpy as np

def QFT(n):
    
    qr = QuantumRegister(n)
    qc = QuantumCircuit(qr)

    for i in range(n - 1, -1, -1):
        
        qc.h(qr[i])
        
        for j in range(i -1, -1, -1):
        
            rotation = i - j + 1
            
            qc.cp(2 * np.pi / 2**rotation, j, i)
    
    for qubit in range(n // 2):
        qc.swap(qubit, n - qubit - 1)

    return qc



def InverseQFT( n):
    qr = QuantumRegister(n)
    circuit = QuantumCircuit(qr)
    qft_circ = QFT(n)
    invqft_circ = qft_circ.inverse()
    circuit.append(invqft_circ, circuit.qubits[:n])
    return circuit.decompose() 


qc = QFT(2)
qc1 = InverseQFT(2)
backend = Aer.get_backend('unitary_simulator')
job = backend.run(qc)
result = job.result()
# Show the results
matrix = result.get_unitary(qc, decimals=3)

print(matrix)

print('\n')

job = backend.run(qc1)
result = job.result()
# Show the results
matrix = result.get_unitary(qc1, decimals=3)

print(matrix)