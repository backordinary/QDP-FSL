# https://github.com/armorsun/AmplifiQation/blob/32059a7e6648f5bc8994032b4664a2f45dfb2d5b/backend/qft.py
"""
Author: AmplifiQation

iQuHACK 2023

Sourced from https://qiskit.org/textbook/ch-algorithms/quantum-fourier-transform.html#generalqft.
"""
from math import pi
from qiskit import QuantumCircuit


def qft_rotations(circuit: QuantumCircuit, n: int):
    """
    Performs qft on the first n qubits in circuit (without swaps)
    param circuit: QuantumCircuit
    param n: int
        Number of qubits.
    return None
        Operation is done in place.
    """
    if n == 0:
        return circuit
    n -= 1
    circuit.h(n)
    for qubit in range(n):
        circuit.cp(pi/2**(n-qubit), qubit, n)
    # At the end of our function, we call the same function again on
    # the next qubits (we reduced n by one earlier in the function)
    qft_rotations(circuit, n)


def swap_registers(circuit: QuantumCircuit, n: int) -> QuantumCircuit:
    """
    Performs the swaps
    :param circuit: QuantumCircuit
    :param n: int
        Swaps are performed on the first n qubits.
    :return: QuantumCircuit
        Returns the transformed circuit
    """
    for qubit in range(n//2):
        circuit.swap(qubit, n-qubit-1)
    return circuit


def qft(circuit: QuantumCircuit, n: int) -> QuantumCircuit:
    """
    QFT on the first n qubits in circuit.
    :param circuit: QuantumCircuit
    :param n: int
        QFT is performed on the first n qubits
    :return: The transformed circuit.
    """
    qft_rotations(circuit, n)
    swap_registers(circuit, n)
    return circuit
