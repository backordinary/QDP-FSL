# https://github.com/f-fathurrahman/ffr-quantum-computing/blob/519c3e395380bf1602446dd2f000f81e36c6abc2/daniel_koch_lectures/lesson_5_1_v4.py
from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit, execute, Aer
import numpy as np
import math

S_simulator = Aer.backends(name="statevector_simulator")[0]
M_simulator = Aer.backends(name="qasm_simulator")[0]

q = QuantumRegister(2, name="q")
c = ClassicalRegister(2, name="c")
deutsch_qc = QuantumCircuit(q, c, name="qc")

deutsch_qc.id(q[0])
deutsch_qc.x(q[1])

import our_qiskit_functions as oqf

f = oqf.deutsch(deutsch_qc, q)
deutsch_qc.measure(q, c)

job = execute(deutsch_qc, M_simulator, shots=1)
res = job.result()
counts_dict = res.get_counts(deutsch_qc)
print("counts_dict = ", counts_dict)
Qubit0_M = list(counts_dict.keys())[0][1]
print("Qubit0_M = ", Qubit0_M)
if Qubit0_M == "0":
    print("Measured state |0>: therefore f is a constant")
else:
    print("Measured state |1>: therefore f is balanced")

print("\nhidden f: ", f)