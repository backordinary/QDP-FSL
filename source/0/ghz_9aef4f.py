# https://github.com/epiqc/PartialCompilation/blob/50d80f56efdf754e40a0b1dd00404788a03fdf3d/qiskit-terra/examples/python/ghz.py
# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
GHZ state example. It also compares running on experiment and simulator

Note: if you have only cloned the Qiskit repository but not
used `pip install`, the examples only work from the root directory.
"""

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import IBMQ, BasicAer, execute
from qiskit.providers.ibmq import least_busy


###############################################################
# Make a quantum circuit for the GHZ state.
###############################################################
q = QuantumRegister(5, "q")
c = ClassicalRegister(5, "c")
qc = QuantumCircuit(q, c, name='ghz')

# Create a GHZ state
qc.h(q[0])
qc.cx(q[0], q[0+1])
qc.cx(q[1], q[1+1])
qc.cx(q[2], q[2+1])
qc.cx(q[3], q[3+1])
# Insert a barrier before measurement
qc.barrier()
qc.measure(q[0], c[0])
qc.measure(q[1], c[1])
qc.measure(q[2], c[2])
qc.measure(q[3], c[3])
qc.measure(q[4], c[4])

###############################################################
# Set up the API and execute the program.
###############################################################
try:
    IBMQ.load_accounts()
except:
    print("""WARNING: There's no connection with the API for remote backends.
             Have you initialized a file with your personal token?
             For now, there's only access to local simulator backends...""")

# First version: simulator
sim_backend = BasicAer.get_backend('qasm_simulator')
job = execute(qc, sim_backend, shots=1024)
result = job.result()
print('Qasm simulator')
print(result.get_counts(qc))

# Second version: real device
least_busy_device = least_busy(IBMQ.backends(simulator=False,
                                             filters=lambda x: x.configuration().n_qubits > 4))
print("Running on current least busy device: ", least_busy_device)
job = execute(qc, least_busy_device, shots=1024)
result = job.result()
print(result.get_counts(qc))
