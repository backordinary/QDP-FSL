# https://github.com/HHHUUUGGGOOO/Quantum_Final_Project/blob/a0f34081d47290fd711659af683f5987ae576fcd/Alice_Bob.py
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute, Aer
from matplotlib import pyplot as plt
from qiskit.visualization import plot_histogram

import numpy as np

sim_qasm = Aer.get_backend('qasm_simulator')

U1_q = QuantumRegister(2)
U1_c = QuantumCircuit(U1_q, name='U1')
U1 = U1_c.to_gate()
U1_dag = U1.inverse()

def construct_encode(U):
    return U.control(1)

def construct_decode(U):
    circ = QuantumCircuit(3, name=U.name + '_decode')
    circ.x(0)
    circ.append(U.control(1), [0, 1, 2])
    return circ.to_gate()

def construct_alice(U):
    circ = QuantumCircuit(3, name='Alice_' + U.name)
    E = construct_encode(U)
    circ.append(E, [0, 1, 2])
    return circ.to_gate()

def construct_bob(U):
    circ = QuantumCircuit(3, name='Bob_' + U.name)
    D = construct_decode(U)
    circ.append(D, [0, 1, 2])
    return circ.to_gate()

if __name__ == "__main__":
    ka = QuantumRegister(1, name='ka')
    kb = QuantumRegister(1, name='kb')
    m = QuantumRegister(2, name='m')
    c = ClassicalRegister(2)
    qc = QuantumCircuit(ka, kb, m, c)

    # Build EPR pair
    qc.h(ka)
    qc.cnot(ka, kb)
    qc.x(kb)
    qc.z(kb)
    qc.barrier(ka,kb)

    alice = construct_alice(U1)
    bob = construct_bob(U1)

    qc.append(alice, [ka, m[0], m[1]])
    qc.append(bob, [kb, m[0], m[1]])

    qc.measure(m, c)
    qc.draw(output='mpl')

    job = execute(qc, backend=sim_qasm, shots=1e6)
    result = job.result()
    counts = result.get_counts()
    plot_histogram(counts)

    plt.show()
