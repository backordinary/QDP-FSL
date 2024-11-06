# https://github.com/Schwarf/qiskit_fundamentals/blob/eb99ec383ec99e2c7e593125a21f28aa6ee0227e/quantum_circuits/quantum_fourier_transform.py
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi
# importing Qiskit
from qiskit import QuantumCircuit, transpile, assemble, Aer, IBMQ
from qiskit.providers.ibmq import least_busy
from qiskit.tools.monitor import job_monitor
from qiskit.visualization import plot_histogram, plot_bloch_multivector


def rotations_quantum_fourier_trafo(circuit, n):
    if n == 0: # Exit function if circuit is empty
        return circuit
    n -= 1 # Indexes start from 0
    circuit.h(n) # Apply the H-gate to the most significant qubit
    for qubit in range(n):
        # For each less significant qubit, we need to do a
        # smaller-angled controlled rotation:
        rotation_angle = pi/2**(n-qubit)
        circuit.cp(rotation_angle, qubit, n)
    rotations_quantum_fourier_trafo(circuit, n)


def swap_registers(circuit, n):
    for qubit in range(n//2):
        circuit.swap(qubit, n-qubit-1)
    return circuit


def quantum_fourier_trafo(circuit, n):
    """QFT on the first n qubits in circuit"""
    rotations_quantum_fourier_trafo(circuit, n)
    swap_registers(circuit, n)
    return circuit

qc2 = QuantumCircuit(4)
quantum_fourier_trafo(qc2,4)
qc2.draw()

plt.show()