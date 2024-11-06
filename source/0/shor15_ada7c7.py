# https://github.com/alu0101133201/QuantumMicroprograms/blob/fbe2b0ada3d0e6dd0f773f1febe7797e1e8a4e5e/src/shor15.py
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, Aer, transpile, assemble
from qiskit.visualization import plot_histogram
from math import gcd
from numpy.random import randint
import pandas as pd
from fractions import Fraction
from math import gcd

# Returns the controlled U gate which implements the operation 7^power mod 15
def c_amod15(power):
    U = QuantumCircuit(4)        
    for iteration in range(power):
        U.x(range(4))
        U.swap(1, 2)
        U.swap(2, 3)
        U.swap(0, 3)
    U = U.to_gate()
    U.name = "7^%i mod 15" % (power)
    c_U = U.control()
    return c_U

#Returns the Inverse Quantum Fourier Transform for n cúbits
def qft_dagger(n):
    qc = QuantumCircuit(n)
    # Don't forget the Swaps!
    for qubit in range(n//2):
        qc.swap(qubit, n-qubit-1)
    for j in range(n):
        for m in range(j):
            qc.cp(-np.pi/float(2**(j-m)), m, j)
        qc.h(j)
    qc.name = "QFT†"
    return qc

# -------------------------
# Basic data for the circuit 

N = 15
n_count = 8
qc = QuantumCircuit(4+n_count, n_count)

# Construction of the circuit
for q in range(n_count):  # Initializating control cúbits in state |+>
    qc.h(q)     
qc.x(3+n_count) # Target cúbits to state |1>
for q in range(n_count): # Controlled-U operations
    qc.append(c_amod15(2**q), [q] + [i+n_count for i in range(4)])
qc.append(qft_dagger(n_count), range(n_count)) # Inverse-QFT
qc.measure(range(n_count), range(n_count))


# Simulate Results
qasm_sim = Aer.get_backend('qasm_simulator')
# Setting memory=True below allows us to see a list of each sequential reading
t_qc = transpile(qc, qasm_sim)
obj = assemble(t_qc, shots=1)
result = qasm_sim.run(obj, memory=True).result()
readings = result.get_memory()
print("Register Reading: " + readings[0])
phase = int(readings[0],2)/(2**n_count)
print("Corresponding Phase: %f" % phase)

obj = assemble(t_qc, shots=1024)
results = qasm_sim.run(obj).result()
answer = results.get_counts()
plot_histogram(answer)


# Get factors from phase
frac = Fraction(phase).limit_denominator(15)
s, r = frac.numerator, frac.denominator
print(r)

guesses = [gcd(7**(r//2)-1, N), gcd(7**(r//2)+1, N)]
print(guesses)
