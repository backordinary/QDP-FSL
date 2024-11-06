# https://github.com/epiqc/PartialCompilation/blob/50d80f56efdf754e40a0b1dd00404788a03fdf3d/qiskit-terra/examples/python/qft.py
# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Quantum Fourier Transform examples.

Note: if you have only cloned the Qiskit repository but not
used `pip install`, the examples only work from the root directory.
"""

import math
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import execute, BasicAer, IBMQ
from qiskit.providers.ibmq import least_busy


###############################################################
# make the qft
###############################################################
def input_state(circ, q, n):
    """n-qubit input state for QFT that produces output 1."""
    for j in range(n):
        circ.h(q[j])
        circ.u1(math.pi/float(2**(j)), q[j]).inverse()


def qft(circ, q, n):
    """n-qubit QFT on q in circ."""
    for j in range(n):
        for k in range(j):
            circ.cu1(math.pi/float(2**(j-k)), q[j], q[k])
        circ.h(q[j])


q = QuantumRegister(5, "q")
c = ClassicalRegister(5, "c")
qft3 = QuantumCircuit(q, c, name="qft3")
qft4 = QuantumCircuit(q, c, name="qft4")
qft5 = QuantumCircuit(q, c, name="qft5")

input_state(qft3, q, 3)
qft3.barrier()
qft(qft3, q, 3)
qft3.barrier()
qft3.measure(q[0], c[0])
qft3.measure(q[1], c[1])
qft3.measure(q[2], c[2])

input_state(qft4, q, 4)
qft4.barrier()
qft(qft4, q, 4)
qft4.barrier()
qft4.measure(q[0], c[0])
qft4.measure(q[1], c[1])
qft4.measure(q[2], c[2])
qft4.measure(q[3], c[3])

input_state(qft5, q, 5)
qft5.barrier()
qft(qft5, q, 5)
qft5.barrier()
qft5.measure(q[0], c[0])
qft5.measure(q[1], c[1])
qft5.measure(q[2], c[2])
qft5.measure(q[3], c[3])
qft5.measure(q[4], c[4])

print(qft3)
print(qft4)
print(qft5)

###############################################################
# Set up the API and execute the program.
###############################################################
try:
    IBMQ.load_accounts()
except:
    print("""WARNING: There's no connection with the API for remote backends.
             Have you initialized a file with your personal token?
             For now, there's only access to local simulator backends...""")

print('Qasm simulator')
sim_backend = BasicAer.get_backend('qasm_simulator')
job = execute([qft3, qft4, qft5], sim_backend, shots=1024)
result = job.result()
print(result.get_counts(qft3))
print(result.get_counts(qft4))
print(result.get_counts(qft5))

# Second version: real device
least_busy_device = least_busy(IBMQ.backends(simulator=False,
                                             filters=lambda x: x.configuration().n_qubits > 4))
print("Running on current least busy device: ", least_busy_device)
job = execute([qft3, qft4, qft5], least_busy_device, shots=1024)
result = job.result()
print(result.get_counts(qft3))
print(result.get_counts(qft4))
print(result.get_counts(qft5))

