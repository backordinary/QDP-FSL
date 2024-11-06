# https://github.com/f-fathurrahman/ffr-quantum-computing/blob/2ee318932a9c8b395993e10dfe4d9f2cb1d52c28/daniel_koch_lectures/lesson_01_04.py
# Test Hadamard gate, 2 qubits

from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit, execute, Aer
import numpy as np
import math

S_simulator = Aer.backends(name="statevector_simulator")[0]
M_simulator = Aer.backends(name="qasm_simulator")[0]

q = QuantumRegister(2)
H_circuit = QuantumCircuit(q)

H_circuit.h(q[0])
H_circuit.h(q[1])

import our_qiskit_functions as oqf
oqf.Wavefunction(H_circuit)

