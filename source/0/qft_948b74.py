# https://github.com/evercodes/Qbeer_hackathon/blob/b069559d851ca2a278f4aa56037be3364032f21c/algorithms/qft.py
"""
Quantum Fourier Transform examples.
"""

import math
from qiskit import QuantumCircuit
from qiskit import execute, BasicAer

def input_state(circ, n):
    """n-qubit input state for QFT that produces output 1."""
    for j in range(n):
        circ.h(j)
        circ.p(-math.pi / float(2 ** (j)), j)


def qft(circ, n):
    """n-qubit QFT on q in circ."""
    for j in range(n):
        for k in range(j):
            circ.cp(math.pi / float(2 ** (j - k)), j, k)
        circ.h(j)


qft3 = QuantumCircuit(5, 5, name="qft3")
qft4 = QuantumCircuit(5, 5, name="qft4")
qft5 = QuantumCircuit(5, 5, name="qft5")

input_state(qft3, 3)
qft3.barrier()
qft(qft3, 3)
qft3.barrier()
qft3.measure(0, 0)
qft3.measure(1, 1)
qft3.measure(2, 2)

input_state(qft4, 4)
qft4.barrier()
qft(qft4, 4)
qft4.barrier()
qft4.measure(0, 0)
qft4.measure(1, 1)
qft4.measure(2, 2)
qft4.measure(3, 3)

input_state(qft5, 5)
qft5.barrier()
qft(qft5, 5)
qft5.barrier()
qft5.measure(0, 0)
qft5.measure(1, 1)
qft5.measure(2, 2)
qft5.measure(3, 3)
qft5.measure(4, 4)

print(qft3)
print(qft4)
print(qft5)

print("Qasm simulator")
sim_backend = BasicAer.get_backend("qasm_simulator")
job = execute([qft3, qft4, qft5], sim_backend, shots=1024)
result = job.result()
print(result.get_counts(qft3))
print(result.get_counts(qft4))
print(result.get_counts(qft5))
