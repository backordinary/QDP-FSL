# https://github.com/parasol4791/quantumComp/blob/0a9a56334d10280e86376df0d57fdf8f4051093d/algos/quantum_phase_estimate.py
# Quantum phase estimate (QPE) algo

#initialization
import matplotlib.pyplot as plt
import numpy as np
import math
from math import pi

# importing Qiskit
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister

# import basic plot tools
from qiskit.visualization import plot_histogram

from qft import qft_dagger_append
from utils.backends import get_job_aer, get_job_ibmq

def qpe(n, eigenstate=1, gate_type='CP', gate_angle=pi/4.):
    """Creates a circuit to estimate quantum phase.
        n - number of measuring qubits. An extra qybit representing an eigenstate will be added.
        eigenstate - we estimate a phase (eigenvalue) of the Unitary operator w.r.t. this state.
        gate_type - type of controlled gate to represent the Unitary operator.
        gate_angle - rotational angle of the controlled gate, if applicable"""
    qpe_qc = QuantumCircuit(n + 1, n)
    # H-gates for all measuring qubits
    for q in range(n):
        qpe_qc.h(q)

    # Initialize eigenvector state
    if eigenstate == 1:
        qpe_qc.x(n)

    # Add CU gates. In Qiskit the order of qubits is reversed
    repetitions = 1
    for counting_qubit in range(n):
        for i in range(repetitions):
            if gate_type == 'CP':
                qpe_qc.cp(gate_angle, counting_qubit, n)  # This is CU
            elif gate_type == 'CRZ':
                qpe_qc.crz(gate_angle, counting_qubit, n)  # This is CU
            else:
                raise ValueError(f"Unsupported gate type {gate_type}")
        repetitions *= 2

    # Append an inverse QFT to the first n qubits of the circuit
    qpe_qc = qft_dagger_append(qpe_qc, n)

    # Measure first n qubits
    for q in range(n):
        qpe_qc.measure(q, q)
    return qpe_qc


def getProbabilities(counts):
    """Returns a map of decimal numbers with probabilities"""
    shots = sum(counts.values())
    return { int(bNum, 2) : ct/shots for bNum, ct in counts.items()}

def getAverage(counts):
    """Returns an average number from all counts"""
    probs = getProbabilities(counts)
    return sum([num * prob for num, prob in probs.items()])

def getMax(counts):
    """Returns a number with maximum probability from all counts"""
    maxCt = -1
    maxNum = -1
    for bNum, ct in counts.items():
        if ct > maxCt:
            maxCt = ct
            maxNum = bNum
    return int(maxNum, 2) # convert from binary to decimal

def printResults(counts, n, cu_gate_angle):
    """Prints results, given a map of measured values with their counts"""
    print(f"n = {n}")
    print(f"CU agle: {cu_gate_angle}")
    expected_theta = cu_gate_angle / (2 * pi)
    print(f"Expected Theta = {expected_theta}")
    print(f"Counts: {counts}")
    maxNum = getMax(counts)
    max_theta = maxNum / (2 ** n)
    diffMax = abs(expected_theta - max_theta) / expected_theta * 100.
    print(f"Nax number = {maxNum} ({bin(maxNum)[2:]})")
    print(f'Max value Theta = {max_theta}, Diff = {round(diffMax, 4)} %')
    average_theta = getAverage(counts) / (2 ** n)
    diffAver = abs(expected_theta - average_theta) / expected_theta * 100.
    print(f'Average Theta = {average_theta}, Diff = {diffAver} %')


if __name__ == "__main__":
    n = 3 # number of measuring qubits
    eigenstate = 1  # eigenvector to the unitary matrix
    # control gate + rotation angle
    # Conptrolled phase
    gate_type = 'CRZ'
    #gate_angle = pi / 4
    gate_angle = 2 * pi / 3

    qpe_qc = qpe(n, eigenstate, gate_type, gate_angle)
    qpe_qc.draw()

    # Simulation
    job = get_job_aer(qpe_qc, shots=2048)
    counts = job.result().get_counts()
    print('Simulation')
    printResults(counts, n, gate_angle)

    # Expected Theta = cu_andle / (2 * pi)
    # We measure a number k in the range [1, 2^n]
    # Theta = x / 2^n

    # n = 3
    # CU andle = pi/4
    # Expected Theta = pi / (4 * 2 * pi) = 1/8
    # Simulation counts: {'001': 1024}
    # In this case, x = 1. (001), and 2^n = 8, so Theta = 1/8

    # n = 3
    # CU andle = 2 * pi/3
    # Expected Theta = 2 *pi / (3 * 2 * pi) = 1/3
    # Simulation counts: {'010': 159, '111': 8, '011': 704, '000': 21, '101': 19, '001': 40, '110': 16, '100': 57}
    # In this case, x = 3. (highest count of 011), and 2^n = 8, so Theta = 3/8 - an approximation of the expected value of 1/3


    # AUTO OUTPUT with AVERAGES

    # n = 3
    # CU agle: 0.7853981633974483 (pi/4)
    # Expected Theta = 0.125
    # Simulation counts: {'001': 2048}
    # Max value Theta = 0.125, Diff = 0.0 %
    # Average Theta = 0.125, Diff = 0.0 %

    # n = 3
    # CU agle: 2.0943951023931953  (2/3 * pi)
    # Expected Theta = 0.3333333333333333
    # Simulation counts: {'101': 34, '010': 370, '110': 22, '100': 115, '000': 39, '011': 1383, '111': 30, '001': 55}
    # Max value Theta = 0.375, Diff = 12.5 %
    # Average Theta = 0.361083984375, Diff = 8.325195312500005 %

    # n = 6
    # CU agle: 2.0943951023931953  (2/3 * pi)
    # Expected Theta = 0.3333333333333333
    # Simulation counts: {'000110': 1, '110000': 1, '100111': 1, '001011': 6, '110011': 1, '111100': 2, '001001': 3, '000111': 1, '101100': 1, '011101': 7, '100000': 4, '010100': 78, '011000': 21, '100011': 1, '010111': 61, '010101': 1398, '010000': 4, '011110': 3, '011100': 4, '010110': 344, '111111': 1, '100010': 3, '001111': 3, '010010': 18, '000101': 3, '010011': 27, '010001': 16, '000100': 1, '100101': 1, '100001': 1, '001010': 2, '000010': 1, '011011': 7, '110110': 1, '011001': 7, '001101': 2, '101000': 1, '001110': 1, '011010': 8, '101010': 1, '111110': 1}
    # Max value Theta = 0.328125, Diff = 1.5625 %
    # Average Theta = 0.33319854736328125, Diff = 0.04043579101561945 %


    # Quantum device
    qpe_qc = qpe(n, eigenstate, gate_type, gate_angle)
    qpe_qc.draw()
    job = get_job_ibmq(qpe_qc, shots=2048)
    counts = job.result().get_counts()
    print('\nQuantum')
    printResults(counts, n, gate_angle)