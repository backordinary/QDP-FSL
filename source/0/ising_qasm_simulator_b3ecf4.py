# https://github.com/abdpdn/Qisikit-Intro-Ising-Model/blob/7744021d2d24b62f8f20f5b6cf4485af19bb769a/ising_qasm_simulator.py
from qiskit import QuantumCircuit, execute, Aer, IBMQ, QuantumRegister
from qiskit.providers.aer import QasmSimulator, StatevectorSimulator, UnitarySimulator
from qiskit.compiler import transpile, assemble
from qiskit.tools.monitor import job_monitor
import matplotlib.pyplot as plt
from qiskit.tools.jupyter import *
from qiskit.visualization import *
from qiskit.quantum_info import *
import numpy as np


def thetak(k, lamb):
    num = lamb - np.cos(2 * np.pi * k / n)
    denom = np.sqrt((lamb - np.cos(2 * np.pi * k / n)) ** 2 + np.sin(2 * np.pi * k / n) ** 2)
    theta = np.arccos(num / denom)
    return theta


#Create functions based on the decomposition included in appendix of Ising paper
def bog(qcirc, q1, q2, theta):
    qcirc.x(q2)
    qcirc.cx(q2, q1)
    #Controlled RX gate
    qcirc.rz(np.pi / 2, q2)
    qcirc.ry(theta / 2, q2)
    qcirc.cx(q1, q2)
    qcirc.ry(-theta / 2, q2)
    qcirc.cx(q1, q2)  #changed from qc to qcirc here - Bruna
    qcirc.rz(-np.pi / 2, q2)
    #####################
    qcirc.cx(q2, q1)
    qcirc.x(q2)
    qcirc.barrier()
    return qcirc


def fourier(qcirc, q1, q2, phase):
    qcirc.rz(phase, q1)
    qcirc.cx(q1, q2)
    #Controlled Hadamard
    qcirc.sdg(q1)
    qcirc.h(q1)
    qcirc.tdg(q1)
    qcirc.cx(q2, q1)
    qcirc.t(q1)
    qcirc.h(q1)
    qcirc.s(q1)
    ####################
    qcirc.cx(q1, q2)
    qcirc.cz(q1, q2)
    qcirc.barrier()
    return qcirc


def digit_sum(n):
    num_str = str(n)
    sum = 0
    for i in range(0, len(num_str)):
        sum += int(num_str[i])
    return sum


def ground_state(lamb):  # backend is now an imput, so we can plot 
    # different ones easily - Bruna
    qc = QuantumCircuit(4, 4)
    #Set correct ground state if lambda < 1
    if lamb < 1:
        qc.x(3)
        qc.barrier()

    #Apply disentangling gates
    qc = bog(qc, 0, 1, thetak(1., lamb))
    qc = fourier(qc, 0, 1, 2 * np.pi / n)
    qc = fourier(qc, 2, 3, 0.)
    qc = fourier(qc, 0, 1, 0.)
    qc = fourier(qc, 2, 3, 0.)
    qc.measure(0, 0)
    qc.measure(1, 1)
    qc.measure(2, 2)
    qc.measure(3, 3)

    backend = Aer.get_backend('qasm_simulator')

    shots = 1024
    job = execute(qc, backend=backend, shots=shots)
    job_monitor(job)
    result = job.result()
    counts = result.get_counts(qc)

    r1 = list(counts.keys())
    r2 = list(counts.values())
    M = 0
    for j in range(0, len(r1)):
        M = M + (4 - 2 * digit_sum(r1[j])) * r2[j] / shots
    mag = M / 4
    return mag


n = 4

lmbd = np.arange(0.0, 2.25, 0.25)
sigmaz = []
for l in lmbd:
    print(f'lambda: {l}')
    sigmaz.append(ground_state(l))

print(sigmaz)

np.save(f'data/ising_qasm_simulator_N_{n}.npy', sigmaz)

