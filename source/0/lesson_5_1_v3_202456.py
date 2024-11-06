# https://github.com/f-fathurrahman/ffr-quantum-computing/blob/519c3e395380bf1602446dd2f000f81e36c6abc2/daniel_koch_lectures/lesson_5_1_v3.py
from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit, execute, Aer
import numpy as np
import math

S_simulator = Aer.backends(name="statevector_simulator")[0]
M_simulator = Aer.backends(name="qasm_simulator")[0]

q = QuantumRegister(2, name="q")
deutsch_qc = QuantumCircuit(q, name="qc")

deutsch_qc.h(q[0])
deutsch_qc.x(q[1])
deutsch_qc.h(q[1])

import our_qiskit_functions as oqf

print("\nInitial state")
oqf.Wavefunction(deutsch_qc)

#deutsch_qc.draw()

f = oqf.blackbox_g_D(deutsch_qc, q)

print("After blackbox_g_D:")
oqf.Wavefunction(deutsch_qc)

deutsch_qc.h(q[0])
deutsch_qc.h(q[1])

print("After H^2:")
oqf.Wavefunction(deutsch_qc)

