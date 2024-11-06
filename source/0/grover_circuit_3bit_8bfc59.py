# https://github.com/francienbarkhof/Quantum_Group1_Project/blob/534e7d4b9c6805b7dcc4dae677869e1a563e391b/grover_circuit_3bit.py
#initialization
import matplotlib.pyplot as plt
import numpy as np

# importing Qiskit
from qiskit import IBMQ, Aer, assemble, transpile
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.providers.ibmq import least_busy
from qiskit import execute
from qiskit.visualization import *
from quantuminspire.qiskit import QI


def initialize_s(qc, qubits):
    """Apply a H-gate to 'qubits' in qc"""
    for q in qubits:
        qc.h(q)
    return qc

if __name__ == '__main__':
    sim = Aer.get_backend('aer_simulator')

    # qi_backend = QI.get_backend('QX single-node simulator')

    n = 3
    grover_circuit = QuantumCircuit(n)
    grover_circuit = initialize_s(grover_circuit, [0,1,2])

    # Oracle
    grover_circuit.barrier
    grover_circuit.cz(0,1) 

    # Diffusion operator (U_s)
    grover_circuit.barrier
    grover_circuit.h([0,1,2])
    grover_circuit.x([0,1,2])
    grover_circuit.ccz(0,1,2)
    grover_circuit.x([0,1,2])
    grover_circuit.h([0,1,2])
    grover_circuit.barrier


    #grover_circuit.draw(output='mpl')

    #q_obj = execute(grover_circuit, backend=qi_backend, shots=256)
    # q_result = q_obj.restult()
    # histogram = q_result.get_counts(grover_circuit)
    # plot_histogram(histogram)

    # we need to make a copy of the circuit with the 'save_statevector'
    # instruction to run on the Aer simulator
    grover_circuit_sim = grover_circuit.copy()
    grover_circuit_sim.save_statevector()
    qobj = assemble(grover_circuit_sim)
    result = sim.run(qobj).result()
    statevec = result.get_statevector()

    print(statevec)

    grover_circuit.measure_all()

    print(grover_circuit)

    aer_sim = Aer.get_backend('aer_simulator')
    qobj = assemble(grover_circuit)
    result = aer_sim.run(qobj).result()
    print(result)
    counts = result.get_counts()
    print("counts:")
    print(counts)
    #plot_histogram(counts, color='midnightblue', title="New Histogram")


