# https://github.com/gigglegrovel/IFI_Code/blob/e8ccfd2b2d93e3b72abbf24cb269331370c047dd/QC.py
grover_circuit.append(oracle_ex3, [0,1,2])
grover_circuit.append(diffuser(n), [0,1,2])

#Define Oracle matrix
U = [
     [-1,0,0,0,0,0,0,0],
     [0,1,0,0,0,0,0,0],
     [0,0,1,0,0,0,0,0],
     [0,0,0,1,0,0,0,0],
     [0,0,0,0,-1,0,0,0],
     [0,0,0,0,0,1,0,0],
     [0,0,0,0,0,0,-1,0],
     [0,0,0,0,0,0,0,1]
]

#initialization
import matplotlib.pyplot as plt
import numpy as np

# importing Qiskit
from qiskit import IBMQ, Aer, assemble, transpile
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.providers.ibmq import least_busy

# import basic plot tools
from qiskit.visualization import plot_histogram

grover_circuit = QuantumCircuit(3)
def initialize_s(qc, qubits):
    """Apply a H-gate to 'qubits' in qc"""
    for q in qubits:
        qc.h(q)
    return qc
grover_circuit = initialize_s(grover_circuit, [0,1,2])

qc = QuantumCircuit(3)
#Transform oracle into quantum circuit
qc.unitary(U,[0,1,2])
qc.draw(output='mpl')

#Make use of cx gates and u3 universal gates to create the whole oracle
trans_qc=transpile(qc,basis_gates = ['cx','u3'])
#create gate
oracle_ex3 = trans_qc.to_gate()
trans_qc.draw(output='mpl')

def diffuser(nqubits):
    qc = QuantumCircuit(nqubits)
    # Apply transformation |s> -> |00..0> (H-gates)
    for qubit in range(nqubits):
        qc.h(qubit)
    # Apply transformation |00..0> -> |11..1> (X-gates)
    for qubit in range(nqubits):
        qc.x(qubit)
    # Do multi-controlled-Z gate
    qc.h(nqubits-1)
    qc.mct(list(range(nqubits-1)), nqubits-1)  # multi-controlled-toffoli
    qc.h(nqubits-1)
    # Apply transformation |11..1> -> |00..0>
    for qubit in range(nqubits):
        qc.x(qubit)
    # Apply transformation |00..0> -> |s>
    for qubit in range(nqubits):
        qc.h(qubit)
    # We will return the diffuser as a gate
    U_s = qc.to_gate()
    U_s.name = "U$_s$"
    return U_s

#initialize circuit
n = 3
grover_circuit = QuantumCircuit(n)
grover_circuit = initialize_s(grover_circuit, [0,1,2])
#add oracle and diffuser
grover_circuit.append(oracle_ex3, [0,1,2])
grover_circuit.append(diffuser(n), [0,1,2])
#measure
grover_circuit.measure_all()
grover_circuit.draw(output='mpl')

#Display results
aer_sim = Aer.get_backend('aer_simulator')
transpiled_grover_circuit = transpile(grover_circuit, aer_sim)
qobj = assemble(transpiled_grover_circuit)
results = aer_sim.run(qobj).result()
counts = results.get_counts()
plot_histogram(counts)