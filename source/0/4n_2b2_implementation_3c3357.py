# https://github.com/StijnW66/Quantum-Project/blob/f9b6abb906e89f70e665dd53b408e07fabab4b7d/src/quantum/period_finding_algorithms/4n%2B2_implementation.py
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, Aer, transpile, assemble
from qiskit.circuit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.visualization import plot_histogram
from math import gcd
from numpy.random import randint
import pandas as pd
from fractions import Fraction
from qiskit.circuit.library import QFT

import time
import sys
sys.path.append(".")

from src.quantum.gates.controlled_U_a_gate import c_U_a_gate

pd.set_option('display.max_rows', 100)



def period_finding(size, a , N):
    control = QuantumRegister(2*size)
    q = QuantumRegister(2*size + 2)
    b = ClassicalRegister(2*size)

    qc = QuantumCircuit(control, q, b)

    for i in range(2*size):
        qc.h(control[i])

    qc.x(q[0])

    for i in range(2*size):
        U_gate = c_U_a_gate(size, a**2**i, N)
        U_gate.name = "%i^%i mod %i" % (a, 2**i, N)
        qc.append(U_gate, [control[i]] + q[:])

    qc.append(QFT(num_qubits=2*size, approximation_degree=0, do_swaps=True, inverse=True, insert_barriers=False, name='iqft'), control)


    for i in range(2*size):
        qc.measure(control[i], b[i])
    return qc

size = 4
n_count = 2*size
a = 4
N = 15

c = period_finding(size, a, N)
print(c.draw())

aer_sim = Aer.get_backend('aer_simulator')
t_qc = transpile(c, aer_sim)
qobj = assemble(t_qc)
results = aer_sim.run(qobj).result()
counts = results.get_counts()

counts = {k: v for k, v in sorted(counts.items(), key=lambda item: item[1])}

print(counts)
rows, measured_phases = [], []
for output in counts:
    decimal = int(output, 2)  # Convert (base 2) string to decimal
    phase = decimal/(2**n_count)  # Find corresponding eigenvalue
    measured_phases.append(phase)
    # Add these values to the rows in our table:
    rows.append([f"{output}(bin) = {decimal:>3}(dec)",
                 f"{decimal}/{2**n_count} = {phase:.2f}"])
# Print the rows in a table
headers=["Register Output", "Phase"]
df = pd.DataFrame(rows, columns=headers)
print(df)

rows = []
for phase in measured_phases:
    frac = Fraction(phase).limit_denominator(N)
    rows.append([phase, f"{frac.numerator}/{frac.denominator}", frac.denominator])
# Print as a table
headers=["Phase", "Fraction", "Guess for r"]
df = pd.DataFrame(rows, columns=headers)
print(df)
time.sleep(1)
print("\a")
time.sleep(1)
print("\a")
time.sleep(1)
print("\a")