# https://github.com/SparshaRay/Competitions/blob/cb6fcdd70881793d2235ca1d0f3910ece225dd7d/SWH-Mojeeto/Q2.py
from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram
import numpy as np

n = int((input()))


def some_function(n):
    def entangle(a, b):
        if b - a == 3:
            qc.cnot(a, a + 1)
            qc.cnot(b, b - 1)
        elif b - a == 2:
            qc.cnot(a, a + 1)
        elif b - a == 1:
            return
        else:
            qc.cnot(a, a + (b - a) // 2)
            qc.cnot(b, a + (b - a) // 2 + 1)
            entangle(a, a + (b - a) // 2)
            entangle(a + (b - a) // 2 + 1, b)

    qc = QuantumCircuit(n)
    qc.h(0)
    qc.cnot(0, n - 1)
    entangle(0, n - 1)

    print((qc.depth()))
    qc.draw(output='mpl', filename='trial 1.png')


some_function(n)
#
# qc.measure_all() # we measure all the qubits
# backend = Aer.get_backend('qasm_simulator') # we choose the simulator as our backend
# counts = execute(qc, backend, shots = 10000).result().get_counts() # we run the simulation and get the counts
# plot_histogram(counts)
# print(counts)
