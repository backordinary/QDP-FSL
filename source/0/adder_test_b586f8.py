# https://github.com/KernalPanik/QC_Optimizer/blob/e49775ec39526568da6f543d4e1f31d24afcbcd8/TestScripts/adder_test.py
'''
This test script contains test methods used to generate and analyze quantum adder circuits.
'''

import math
import numpy as np
import time

import qiskit
from qiskit import Aer, IBMQ, execute
from qiskit import transpile
from qiskit.providers.aer.noise import NoiseModel
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import BasicAer
from qiskit.aqua import QuantumInstance
from qiskit.visualization import plot_histogram

def four_qubit():
    circ = QuantumCircuit(4, 2)
    circ.x(0)
    circ.x(1)
    circ.ccx(0, 1, 2)
    circ.cx(0, 1)
    circ.ccx(1, 2, 3)
    circ.cx(1, 2)
    circ.cx(0, 1)
    circ.measure(2, 0)
    circ.measure(3, 1)

    provider = IBMQ.load_account()
    device = provider.get_backend('ibmq_16_melbourne')
    #device = Aer.get_backend('qasm_simulator')
    trans_circ = transpile(circ, device)

    job = execute(circ, device, shots=1024)
    result = job.result()
    counts = result.get_counts(circ)
    print(counts)
    plot_histogram(counts)
    print("transpiled circ op count:")
    print(trans_circ.count_ops())

def five_qubit():
    circ = QuantumCircuit(5, 2)
    circ.x(0)
    circ.x(1)
    circ.cx(0, 3)
    circ.cx(1, 3)
    circ.cx(2, 3)
    circ.ccx(0, 1, 4)
    circ.ccx(0, 2, 4)
    circ.ccx(1, 2, 4)
    circ.measure(3, 0)
    circ.measure(4, 1)

    provider = IBMQ.load_account()
    device = provider.get_backend('ibmq_16_melbourne')
    #device = Aer.get_backend('qasm_simulator')
    trans_circ = transpile(circ, device)

    job = execute(circ, device, shots=1024)
    result = job.result()
    counts = result.get_counts(circ)
    print(counts)
    plot_histogram(counts)

    print("transpiled circ op count:")
    print(trans_circ.count_ops())

IBMQ.save_account("") # Add your IBMQ API key

four_qubit()
five_qubit()
