# https://github.com/peiyi1/nassc_code/blob/e8fdeed874f36e9b141842d395c5f08f19a0c878/benchmark/suites/grover_n4.py
import qiskit
from qiskit import QuantumCircuit
import math
import numpy as np

def phase_oracle(qc, n, indices_to_mark):
    
    oracle_matrix = np.identity(2**n)
    for i in indices_to_mark:
        oracle_matrix[i,i] = -1
        
    qc.unitary(qiskit.quantum_info.Operator(oracle_matrix), range(n))
def diffuser(n):
    qc = QuantumCircuit(n)
    
    for qubit in range(n):
        qc.h(qubit)
    
    for qubit in range(n):
        qc.x(qubit)
    
    qc.h(n-1)
    qc.mct(list(range(n-1)), n-1) 
    qc.h(n-1)
    
    for qubit in range(n):
        qc.x(qubit)
    
    for qubit in range(n):
        qc.h(qubit)
    
    return qc
def circuits():
    n=4
    indices_to_mark=[15]

    qc=QuantumCircuit(n)

    for qubit in range(n):
        qc.h(qubit)

    iteration_times=math.floor(math.pi/4*math.sqrt(pow(2,n)))
    for iteration in range(iteration_times):
        phase_oracle(qc, n, indices_to_mark)
        qc+=diffuser(n)

    qc.measure_all()

    return qc
