# https://github.com/Scdk/shor/blob/b7dd48604b87d506358f26d9a46169fa048de97c/shor_exemple.py
import numpy as np
import matplotlib.pyplot as plt

import qiskit as qk
from qiskit.algorithms import Shor
from qiskit.utils import QuantumInstance
from qiskit.tools.monitor import job_monitor 
from qiskit.tools.visualization import plot_histogram 

from apitoken import apitoken 

# Algoritmo usando a solução pronta

# backend = qk.Aer.get_backend('qasm_simulator')
# quantum_instance = QuantumInstance(backend, shots=1000)
# shor = Shor(quantum_instance=quantum_instance)
# my_shor = shor.factor(N=15,a=2)
# print(my_shor)

def c_amod15(a, power):
    U = qk.QuantumCircuit(4)
    for interation in range(power):
        U.swap(2,3)
        U.swap(1,2)
        U.swap(0,1)
        for q in range(4):
            U.x(q)
    U=U.to_gate()
    U.name="%i^%i mod 15" %(a,power)
    c_U = U.control()
    return c_U


def qft_dagger(n):
    qc = qk.QuantumCircuit(n)
    for qubit in range(n//2):
        qc.swap(qubit,n-qubit-1)
    for j in range(n):
        for m in range(j):
            qc.cp(-np.pi/float(2**(j-m)),m,j)
        qc.h(j)
    qc.name = "QFT^-1"
    return qc

n_count = 8
a = 2

qc = qk.QuantumCircuit(n_count + 4,n_count)

for q in range(n_count):
    qc.h(q)

qc.x(3+n_count)

for q in range(n_count):
    qc.append(c_amod15(a,2**q), [q]+[i+n_count for i in range(4)])

qc.append(qft_dagger(n_count),range(n_count))

qc.measure(range(n_count),range(n_count))
qc.draw(output='mpl')
# plt.show()

backend = qk.Aer.get_backend('qasm_simulator')
results = qk.execute(qc, backend, shots = 2048).result()
counts = results.get_counts()
plot_histogram(counts)
plt.show()

# 'Number of qubits (12) in circuit-7 is greater than maximum (5) in the coupling_map'

# print(qk.IBMQ.load_account())
# provider = qk.IBMQ.get_provider('ibm-q')
# qcomp = provider.get_backend('ibmq_santiago')
# job = qk.execute(qc, backend=qcomp)
# job_monitor(job)
# result = job.result()
# plot_histogram(result.get_counts(circuit))
# plt.show()
