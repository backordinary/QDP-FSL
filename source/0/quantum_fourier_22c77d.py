# https://github.com/calwoo/qlearn/blob/8ee23033baf69091c944132378f718ea6c4ce950/algos/quantum_fourier.py
import numpy as np
import matplotlib.pyplot as plt

from qiskit import Aer, QuantumCircuit, ClassicalRegister, QuantumRegister, assemble, transpile
from qiskit.visualization import plot_histogram


# recursive qft rotations function
def qft_rots(circuit, n):
    """pre-swap quantum fourier transform"""
    # base case
    if n == 0:
        return circuit
    else:
        n -= 1
        circuit.h(n)
        for q in range(n):
            # apply controlled-rotate
            circuit.cp(np.pi / 2**(n-q), q, n)
        # recursively call qft rotation
        qft_rots(circuit, n)

# swap registers after rotations
def swap_regs(circuit, n):
    for q in range(n//2):
        circuit.swap(q, n-q-1)
    return circuit

# quantum fourier transform
def qft(circuit, n):
    qft_rots(circuit, n)
    circuit.barrier()
    swap_regs(circuit, n)
    return circuit


n = 4
qc = QuantumCircuit(n)

qft(qc, n)
qc.draw(output="mpl")
plt.show()
