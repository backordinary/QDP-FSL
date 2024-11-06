# https://github.com/pablogalaviz/qiskit_demo/blob/a87c49ce52695cedbd96a7f899f50b90d7b1813e/src/Deutsch-Jozsa.py
# initialization
import numpy as np

# importing Qiskit
from matplotlib import pyplot as plt
from qiskit import IBMQ, Aer
from qiskit.providers.aer import AerSimulator
from qiskit.providers.ibmq import least_busy
from qiskit import QuantumCircuit, assemble, transpile

# import basic plot tools
from qiskit.visualization import plot_histogram

from qiskit.test.mock import FakeVigo
device_backend = FakeVigo()

# set the length of the n-bit input string.
n = 5

const_oracle = QuantumCircuit(n+1)

if np.random.randint(2) == 1:
    const_oracle.x(n)

const_oracle.draw('mpl')
plt.show()

balanced_oracle = QuantumCircuit(n+1)

b_str = "101"

# Place X-gates
for qubit in range(len(b_str)):
    if b_str[qubit] == '1':
        balanced_oracle.x(qubit)

# Use barrier as divider
balanced_oracle.barrier()

# Controlled-NOT gates
for qubit in range(n):
    balanced_oracle.cx(qubit, n)

balanced_oracle.barrier()

# Place X-gates
b_str = ""
for qubit, value in enumerate(np.random.randint(2, size=n)):
    b_str+="%d"%value
    if value == 1:
        balanced_oracle.x(qubit)
print(b_str)

balanced_oracle.draw('mpl')
plt.show()

# %%
dj_circuit = QuantumCircuit(n+1, n)

# Apply H-gates
for qubit in range(n):
    dj_circuit.h(qubit)

# Put qubit in state |->
dj_circuit.x(n)
dj_circuit.h(n)
dj_circuit.draw('mpl')
plt.show()

# Add oracle
if np.random.randint(2) == 1:
    print("constant oracle")
    dj_circuit.compose(const_oracle, inplace=True)
else:
    print("balanced oracle")
    dj_circuit.compose(balanced_oracle, inplace=True)

# Repeat H-gates
for qubit in range(n):
    dj_circuit.h(qubit)
dj_circuit.barrier()

# Measure
for i in range(n):
    dj_circuit.measure(i, i)


dj_circuit.draw('mpl')
plt.show()

# %%
# use local simulator
sim_vigo = AerSimulator.from_backend(device_backend)

qobj = assemble(dj_circuit, sim_vigo)
results = sim_vigo.run(qobj).result()
answer = results.get_counts()

plot_histogram(answer)
plt.show()

print(answer)


