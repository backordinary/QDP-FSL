# https://github.com/calwoo/qlearn/blob/8ee23033baf69091c944132378f718ea6c4ce950/algos/bernstein_vazirani.py
import numpy as np
import matplotlib.pyplot as plt

from qiskit import Aer, QuantumCircuit, ClassicalRegister, QuantumRegister, assemble, transpile
from qiskit.visualization import plot_histogram


# num qubits and hidden mask
s = "11101010"
n = len(s)

# circuit that implements x `dot` s `mod` 2
f = QuantumCircuit(n+1)

s = s[::-1]
for q in range(n):
    if s[q] == "1":
        f.cx(q, n)

# bernstein-vazirani circuit
bv = QuantumCircuit(n+1, n)

# set ancilla into |-> state
bv.x(n)
bv.h(n)

# construct superposition state
for q in range(n):
    bv.h(q)

bv.barrier()

# apply oracle
bv += f
bv.barrier()

# undo hadamard and measure
for q in range(n):
    bv.h(q)

for q in range(n):
    bv.measure(q, q)

bv.draw(output="mpl")
plt.show()

# simulate!
sim = Aer.get_backend("qasm_simulator")
qobj = assemble(bv, sim)
results = sim.run(qobj).result()
answer = results.get_counts()

plot_histogram(answer)
plt.show()
