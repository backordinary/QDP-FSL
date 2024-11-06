# https://github.com/f-fathurrahman/ffr-quantum-computing/blob/2ee318932a9c8b395993e10dfe4d9f2cb1d52c28/daniel_koch_lectures/lesson_01_01.py
from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit, execute, Aer
import numpy as np
import math

S_simulator = Aer.backends(name="statevector_simulator")[0]
M_simulator = Aer.backends(name="qasm_simulator")[0]

q = QuantumRegister(1)
hello_qubit = QuantumCircuit(q)

hello_qubit.id(q[0])

job = execute(hello_qubit, S_simulator)
result = job.result()

print(result.get_statevector())