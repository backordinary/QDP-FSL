# https://github.com/dsaf2007/QuantumComputing/blob/b973d39f51f2653bff60658ab86c632f45c3f32b/project2/grover_2qubit.py
#initialization
import matplotlib.pyplot as plt
import numpy as np

# importing Qiskit
from qiskit import IBMQ, Aer, assemble, transpile
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.providers.ibmq import least_busy

# import basic plot tools
from qiskit.visualization import plot_histogram

from qiskit_textbook.tools import vector2latex

#two qubits
n = 2
grover_circuit = QuantumCircuit(n)

#fuction to initialize s


def initialize_s(qc, qubits):
    """Apply a H-gate to 'qubits' in qc"""
    for q in qubits:
        qc.h(q)
    return qc

grover_circuit = initialize_s(grover_circuit, [0,1])
grover_circuit.draw(output='mpl',filename='circuit.png')

#apply Oracle for |w> = |11>
grover_circuit.cz(0,1) # Oracle
grover_circuit.draw(output='mpl',filename='oracle.png')

#apply diffuser
# Diffusion operator (U_s)
grover_circuit.h([0,1])
grover_circuit.z([0,1])
grover_circuit.cz(0,1)
grover_circuit.h([0,1])
grover_circuit.draw(output='mpl',filename='diffuser.png')

#run circuit in simmulation
sim = Aer.get_backend('aer_simulator')
# we need to make a copy of the circuit with the 'save_statevector'
# instruction to run on the Aer simulator
grover_circuit_sim = grover_circuit.copy()
grover_circuit_sim.save_statevector()
qobj = assemble(grover_circuit_sim)
result = sim.run(qobj).result()
statevec = result.get_statevector()
vector2latex(statevec, pretext="|\\psi\\rangle =")

grover_circuit.measure_all()

aer_sim = Aer.get_backend('aer_simulator')
qobj = assemble(grover_circuit)
result = aer_sim.run(qobj).result()
counts = result.get_counts()
plot_histogram(counts,filename='histogram_2qubits.png')