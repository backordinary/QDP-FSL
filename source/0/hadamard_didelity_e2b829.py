# https://github.com/MaxVakili/Quantum-Computing-Qiskit/blob/6636cb5f8c6d8e3ebf37ead58222eff76e3e3384/Hadamard-Didelity.py
#The following program Measures the fidelity of Hadamard Gate
#You can run it on a noisy simulator or an actual machine.

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from numpy import pi

qreg_q = QuantumRegister(1, 'q')
creg_c = ClassicalRegister(1, 'c')
circuit = QuantumCircuit(qreg_q, creg_c)

circuit.h(qreg_q[0])
circuit.h(qreg_q[0])
circuit.measure(qreg_q[0], creg_c[0])