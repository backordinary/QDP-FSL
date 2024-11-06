# https://github.com/stephenhky/QLinear/blob/b5483accb791fb206a652c3d9f133532ef864af3/qlinear/qft.py

from math import pi

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister


def qft_gate(nbdigits):
    qr = QuantumRegister(nbdigits, 'qr')
    circuit = QuantumCircuit(qr)
    for i in range(nbdigits):
        circuit.h(qr[i])
        for j in range(i+1, nbdigits):
            circuit.cp(2*pi/2**(j-i+1), j, i)
    for i in range(nbdigits // 2):
        circuit.swap(i, nbdigits-i-1)
    return circuit.to_gate()


def inv_qft_gate(nbdigits):
    qr = QuantumRegister(nbdigits, 'qr')
    circuit = QuantumCircuit(qr)
    for i in range(nbdigits // 2):
        circuit.swap(i, nbdigits-i-1)
    for i in range(nbdigits):
        circuit.h(qr[i])
        for j in range(i+1, nbdigits):
            rot_phase = - 2*pi/2**(j-i+1)
            circuit.cp(rot_phase, j, i)
    return circuit.to_gate()


def phase_estimation_gate(gate, nbdigits, teststate_nbdigits):
    # customized controlled gate
    controlled_gate = gate.control(1)

    # preparing for the circuit
    circuit = QuantumCircuit(nbdigits + teststate_nbdigits)
    circuit.h(range(nbdigits))
    for i in range(nbdigits):
        for _ in range(2 ** (i)):
            circuit.append(controlled_gate, [i] + [j + nbdigits for j in range(teststate_nbdigits)])
    circuit.append(inv_qft_gate(nbdigits), range(nbdigits))

    return circuit.to_gate()


def inv_phase_estimation_gate(gate, nbdigits, teststate_nbdigits):
    # customized controlled gate
    controlled_gate = gate.control(1)

    # preparing for the circuit
    circuit = QuantumCircuit(nbdigits + teststate_nbdigits)
    circuit.append(qft_gate(nbdigits), range(nbdigits))
    for i in range(nbdigits+1, 0, -1):
        for _ in range(2 ** (i)):
            circuit.append(controlled_gate, [i] + [j + nbdigits for j in range(teststate_nbdigits)])
    circuit.h(range(nbdigits))

    return circuit.to_gate()


def phase_estimation_circuit(gate, nbdigits, teststate_nbdigits, initial_monopartite_state=np.array([0, 1])):
    # initial_state
    assert len(initial_monopartite_state) == 2 ** teststate_nbdigits
    assert np.linalg.norm(initial_monopartite_state) == 1
    initial_state = np.zeros(2 ** (nbdigits + teststate_nbdigits), dtype=np.complex_)
    for i in range(2 ** teststate_nbdigits):
        initial_state[i * 2 ** nbdigits] = initial_monopartite_state[i]

    # building the circuit
    circuit = QuantumCircuit(nbdigits + teststate_nbdigits, nbdigits)
    circuit.initialize(initial_state)
    circuit.append(
        phase_estimation_gate(gate, nbdigits, teststate_nbdigits),
        range(nbdigits + teststate_nbdigits)
    )
    circuit.measure(range(nbdigits), range(nbdigits))

    return circuit


def evolution_operator(A):
    assert np.all(A == np.conj(A.T))
    eigvals, eigvecs = np.linalg.eig(A)
    U = np.diag(np.exp(2 * pi * (1.j) * eigvals))
    U = np.dot(np.dot(eigvecs, U), eigvecs.T)
    return U
    # scipy.linalp.expm
