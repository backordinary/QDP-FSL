# https://github.com/JacobViertel/Biped/blob/02ad195a4f7469e27f418cb2b127d6387a1f30a7/scripts/qiskit/qpe_mood.py
#!/usr/bin/env python
# coding: utf-8

#initialization
import numpy as np
import math
import csv
import sys
import os
# importing Qiskit
from qiskit import IBMQ, Aer
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, execute

pi = np.pi


if os.path.exists("/home/biped/catkin_ws/src/jacob/scripts/results/qpe_mood.csv"):
    os.remove("/home/biped/catkin_ws/src/jacob/scripts/results/qpe_mood.csv")
else:
    pass

# Defining quantum fourier transformation:
def qft_dagger(circ, n):
    """n-qubit QFTdagger the first n qubits in circ"""
    # Don't forget the Swaps!
    for qubit in range(n//2):
        circ.swap(qubit, n-qubit-1)
    for j in range(n):
        for m in range(j):
            circ.cu1(-math.pi/float(2**(j-m)), m, j)
        circ.h(j)


combined = sys.argv[1]
def split(word):
    return [char for char in word]
# print(split(combined)) 
mood = int(combined[0])
intensity = int(combined[1])
eigenstate = 0

qpe = QuantumCircuit(4, 3)
qpe.x(3)
qpe.h(0)
qpe.h(1)
qpe.h(2)
repetitions = 1

if mood == 1:
    eigenstate += math.pi
if intensity == 1:
    eigenstate += math.pi/4
if intensity == 2:
    eigenstate += math.pi/2
for i in range(repetitions):
    qpe.cu1(eigenstate, 0, 3); # This is C-U
repetitions *= 2
for i in range(repetitions):
    qpe.cu1(eigenstate, 1, 3); # This is C-U
repetitions *= 2
for i in range(repetitions):
    qpe.cu1(eigenstate, 2, 3); # This is C-U
repetitions *= 2
# Apply inverse QFT
qft_dagger(qpe, 3)
# Measure
qpe.barrier()
qpe.measure(0,0)
qpe.measure(1,1)
qpe.measure(2,2)


backend = Aer.get_backend('qasm_simulator')
shots = 1
results = execute(qpe, backend=backend, shots=shots).result()
answer = results.get_counts()

mood_result = open("/home/biped/catkin_ws/src/jacob/scripts/results/qpe_mood.csv", "w")
writer = csv.writer(mood_result)
for key, value in answer.items():
    writer.writerow(answer.keys())
mood_result.close()
# print(answer)
print("QPE is done!")
