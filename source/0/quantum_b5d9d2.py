# https://github.com/nkpro2000/IVyearProject/blob/678af0a893b9d6d11af0d05f0babab331f517ee5/quantum.py
import qiskit_
from bits_nums import *

import matplotlib.pyplot as plt

N = 5 # max no. of qbits we have in IBMQ free

def get_qcir(n=None):
    if n is None:
        n = N

    q, c = QuantumRegister(n), ClassicalRegister(n)
    qcir = QuantumCircuit(q, c)
    qcir.h(q)

    # insert qbit manipulate circuit


    qcir.measure(q, c)

    return qcir

def show(qcir):
    qcir.draw(output='mpl').show()

def output(qcir, backend=None):
    if backend == 1:
        backend = qiskit_.qbackend1
    elif backend == 2:
        backend = qiskit_.qbackend2
    else:
        backend = qiskit_.sbackend
    
    job = execute(qcir, backend)

    return result(job, 0)

def plot(counts):
    if type(counts) is not dict:
        counts = counts.get_counts()
    plt.figure("Result")
    plt.plot(counts.keys(), counts.values(), 'X')
    plt.show(block=False)
