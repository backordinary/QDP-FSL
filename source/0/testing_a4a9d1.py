# https://github.com/CAholder/TKET-Testing/blob/4c27af305968f0169ebdd91c697392d38f8140a7/Testing.py
# provider = IBMQ.load_account()
# backend = provider.get_backend('ibmq_lima')

"""
Grover's Search Benchmark Program - Qiskit
"""

import sys
import time
from qiskit import *
from qiskit import IBMQ, Aer
from qiskit.visualization import plot_histogram
from qiskit.providers.aer.noise import NoiseModel

import numpy as np
from pytket.extensions.qiskit import qiskit_to_tk

import pytket.extensions.qsharp as AB

qc = QuantumCircuit(2,2)
qc.h(0)
qc.cx(0,1)

meas = QuantumCircuit(2,2)
meas.measure([0,1],[0,1])

backend = BasicAer.get_backend('qasm_simulator')
drawing = qc.compose(meas)
result = backend.run(transpile(drawing, backend), shots=1000).result()
counts = result.get_counts(qc)
print("Qiskit Measurement: ", counts)

# Measurement pre, pro measurement
tketCirc = qiskit_to_tk(qc)
tketCirc.measure_all()
print(tketCirc)
print(type(tketCirc))

IONQ = AB.AzureBackend("ionq.simulator", resourceId = "/subscriptions/7ef9c9ba-198a-420f-a70b-c4341dd14797/resourceGroups/AzureQuantum/providers/Microsoft.Quantum/Workspaces/IONQTest1forChoi", location="East US", storage="aqa259c256c06c4eac875c34")
# print(IONQ.available_devices())
IONQ.default_compilation_pass()
print("Pass 1")
valid = IONQ.valid_circuit(tketCirc)
print(valid)
# compiled_cric = IONQ.get_compiled_circuit(tketCirc, 0)
# print(compiled_cric)
print("Pass 2")
handle = IONQ.process_circuit(tketCirc, n_shots=5, valid_check=True)
print("Pass 3")
tketResult = IONQ.get_counts(handle)
print(tketResult)