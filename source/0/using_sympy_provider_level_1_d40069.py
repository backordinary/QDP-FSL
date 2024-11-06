# https://github.com/yaelbh/qiskit-sympy-provider/blob/57d2f3e1e95fedd8e5a771084e2cb538b0e18401/examples/using_sympy_provider_level_1.py
# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Example showing how to use the Sympy Provider at level 1 (intermediate).

This example shows how an intermediate user interacts with the Sympy Provider. It builds some circuits
and compiles them. It makes a qobj object which is just a container to be 
run on a backend. The same qobj can run on many backends (as shown). It is the
user responsibility to make sure it can be run. This is useful when you want to compare the same
circuits on different backends or change the compile parameters.
"""

import time

# Import the Qiskit modules
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import compile
from qiskit_addon_sympy import SympyProvider

SyQ = SympyProvider()
    
# Create a quantum and classical register.
qubit_reg = QuantumRegister(2, name='q')
clbit_reg = ClassicalRegister(2, name='c')

# Making first circuit: Bell state
qc1 = QuantumCircuit(qubit_reg, clbit_reg, name="bell")
qc1.h(qubit_reg[0])
qc1.cx(qubit_reg[0], qubit_reg[1])

# Making another circuit: superpositions
qc2 = QuantumCircuit(qubit_reg, clbit_reg, name="superposition")
qc2.h(qubit_reg)

# Setting up the backend
print("(Sympy Backends)")
for backend in SyQ.backends():
    print(backend.status())

statevector_backend = SyQ.get_backend('statevector_simulator')
unitary_backend = SyQ.get_backend('unitary_simulator')

# Compiling the qobj for the statevector backend 
qobj = compile([qc1, qc2], backend=statevector_backend)

# Running the both backends on the same qobj
statevector_job = statevector_backend.run(qobj)
unitary_job = unitary_backend.run(qobj)

lapse = 0
interval = 0.01
while statevector_job.status().name != 'DONE' or unitary_job.status().name != 'DONE':
    print('Status at {} milliseconds'.format(1000 * interval * lapse))
    print("Stevector simulator: ", statevector_job.status())
    print("Unitary simulator: ", unitary_job.status())
    time.sleep(interval)
    lapse += 1

print(statevector_job.status())
print(unitary_job.status())

statevector_result = statevector_job.result()
unitary_result = unitary_job.result()

# Show the results
print("Stevector simulator: ", statevector_result)
print(statevector_result.get_statevector(qc1))
print(statevector_result.get_statevector(qc2))
print("Unitary simulator: ", unitary_result)
print(unitary_result.get_unitary(qc1))
print(unitary_result.get_unitary(qc2))


