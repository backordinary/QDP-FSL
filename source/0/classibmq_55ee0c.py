# https://github.com/AlbertoVari/SolidQML/blob/530cc04a058266c7a5aeaab1795ba14001784d58/classIBMQ.py
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.autograd import Function
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import qiskit
from qiskit import transpile, assemble
from qiskit.visualization import *

from qiskit import QuantumCircuit, execute, Aer, IBMQ, QuantumRegister, ClassicalRegister
from qiskit.compiler import transpile, assemble
from qiskit.tools.jupyter import *
from qiskit.visualization import *
from numpy import pi
import json


IBMQ.load_account()
provider = IBMQ.load_account()
print(provider.backends())
# exit()
#backend = provider.get_backend('ibmq_manila')
backend = provider.get_backend('ibmq_qasm_simulator')
status = backend.status()
is_operational = status.operational
jobs_in_queue = status.pending_jobs
print(is_operational, jobs_in_queue)


class qcircuit:

    def __init__(self, n_qubits):

        all_qubits = [i for i in range(n_qubits)]
        self.theta = np.pi

    def run(self, processor, nshots, angle):

        qreg_q = QuantumRegister(1, 'q')
        creg_c = ClassicalRegister(1, 'c')
        circuit = QuantumCircuit(qreg_q, creg_c)

        circuit.h(qreg_q[0])
        circuit.barrier(qreg_q[0])
        circuit.ry(angle, qreg_q[0])
        circuit.barrier(qreg_q[0])
        circuit.measure(qreg_q[0], creg_c[0])

        job = execute(circuit, processor, shots=nshots)

        result = job.result().get_counts()

        counts = np.array(list(result.values()))
        states = np.array(list(result.keys())).astype(float)

        # Compute probabilities for each state
        probabilities = counts / nshots
        # Get state expectation
        expectation = np.sum(states * probabilities)

        return np.array([expectation])


# simulator = backend
num_shots = 100

trainc = qcircuit(1)
print('Expected value for rotation pi {}'.format(trainc.run(backend, num_shots,np.pi)[0]))

