# https://github.com/iQuHACK/Duckfebugging/blob/73dea9e3390c4566bcc30a4e5690b1c122ef494b/generalized.py

# Importing standard Qiskit libraries and configuring account
from qiskit import QuantumCircuit, execute, Aer, IBMQ
from qiskit.compiler import transpile, assemble
from qiskit.tools.jupyter import *
from qiskit.visualization import *

N = 3
#initializing the circuit
circuit = QuantumCircuit(3*N, 3*N)

def general(circuit, N):
    #hadamards and cnots
    for i in range(N):
        circuit.h(N+i)
        circuit.cx(N+i, 2*N+i)
    circuit.barrier()
    #more cnots and hadamards
    for i in range(N):
        circuit.cx(i, N+i)
        circuit.h(i)
    circuit.barrier()
    #measuring
    for i in range(2*N):
        circuit.measure(i, i)
    circuit.barrier()
    #O(M)
    for i in range(N):
        circuit.cz(i, 2*N+i)
        circuit.cx(N+i, 2*N+i)
        
for i in range(N):
    circuit.measure(2*N+i, 2*N+i)
#simulating
simulator = Aer.get_backend('qasm_simulator')
job = execute(circuit, backend = simulator, shots = 1000)
result = job.result()
print(result.get_counts())