# https://github.com/calwoo/qlearn/blob/8ee23033baf69091c944132378f718ea6c4ce950/algos/deutsch_jozsa.py
import numpy as np
import matplotlib.pyplot as plt

from qiskit import Aer, QuantumCircuit, assemble, transpile
from qiskit.visualization import plot_histogram


# n-bit boolean oracles
n = 3  # f:{0,1}^n -> {0,1}

# constant oracle
const_oracle = QuantumCircuit(n+1)

const_bit = np.random.randint(2)
if const_bit == 1:
    # flip the output bit
    const_oracle.x(n)

# const_oracle.draw(output="mpl")
# plt.show()

# balanced oracle
bal_oracle = QuantumCircuit(n+1)

x_mask = "010"
for i, q in enumerate(x_mask):
    if q == "1":
        bal_oracle.x(i)

bal_oracle.barrier()
for q in range(n):
    bal_oracle.cx(q, n)
bal_oracle.barrier()

# uncompute the X-gates
for i, q in enumerate(x_mask):
    if q == "1":
        bal_oracle.x(i)

# bal_oracle.draw(output="mpl")
# plt.show()


### deutsch-jozsa algorithm circuit
deutsch_jozsa = QuantumCircuit(n+1, n)

# set up superposition states with hadamard gates
for q in range(n):
    deutsch_jozsa.h(q)

# produce an ancilla |-> state for phase kickback
deutsch_jozsa.x(n)
deutsch_jozsa.h(n)

# apply oracle
deutsch_jozsa += bal_oracle

# uncompute hadamard gates
for q in range(n):
    deutsch_jozsa.h(q)
deutsch_jozsa.barrier()

# measure output
for q in range(n):
    deutsch_jozsa.measure(q, q)

deutsch_jozsa.draw(output="mpl")
plt.show()


# simulate
sim = Aer.get_backend("qasm_simulator")
qobj = assemble(deutsch_jozsa, sim)
results = sim.run(qobj).result()
answer = results.get_counts()

plot_histogram(answer)
plt.show()

