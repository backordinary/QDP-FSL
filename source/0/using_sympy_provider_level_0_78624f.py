# https://github.com/yaelbh/qiskit-sympy-provider/blob/57d2f3e1e95fedd8e5a771084e2cb538b0e18401/examples/using_sympy_provider_level_0.py
# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Example showing how to use the Sympy Provider at level 0 (novice).

This example shows the most basic way to user the Sympy Provider. It builds some circuits
and runs them on both the statevector and unitary simulators.
"""

# Import the Qiskit modules
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import execute
from qiskit_addon_sympy import SympyProvider

SyQ = SympyProvider()

# Create a Quantum and Classical Register.
qubit_reg = QuantumRegister(2)
clbit_reg = ClassicalRegister(2)

# making first circuit: bell state
qc1 = QuantumCircuit(qubit_reg, clbit_reg)
qc1.h(qubit_reg[0])
qc1.cx(qubit_reg[0], qubit_reg[1])

# making another circuit: superpositions
qc2 = QuantumCircuit(qubit_reg, clbit_reg)
qc2.h(qubit_reg)

# setting up the backend
print("(Sympy Backends)")
print(SyQ.backends())

# running the statevector simulator
statevector_job = execute([qc1, qc2], SyQ.get_backend('statevector_simulator'))
statevector_result = statevector_job.result()

# show the results
print("Stevector simulator: ", statevector_result)
print(statevector_result.get_statevector(qc1))
print(statevector_result.get_statevector(qc2))

# running the unitary simulator
unitary_job = execute([qc1, qc2], SyQ.get_backend('unitary_simulator'))
unitary_result = unitary_job.result()

# show the results
print("Unitary simulator: ", unitary_result)
print(unitary_result.get_unitary(qc1))
print(unitary_result.get_unitary(qc2))
