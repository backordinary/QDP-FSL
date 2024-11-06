# https://github.com/DanielTongAwesome/Shor-s-Algorithm/blob/8bd993dcaabdfc750f192fda49bbb8e0c8c25a19/main.py
'''
Author: Zitian(Daniel) Tong
Date: 2021-02-24 23:27:44
LastEditTime: 2021-04-14 22:18:16
LastEditors: Zitian(Daniel) Tong
Description: shor's algorithm's main file
FilePath: /Shor's_Algorithm/main.py
'''

from qiskit.aqua.algorithms import Shor
from qiskit.aqua import QuantumInstance
from qiskit import QuantumCircuit, Aer, execute
from qiskit.tools.visualization import plot_histogram
import numpy as np


'''
convert the integer factorization problem into a period finding problem
modular exponantiation function, below function is a hard-coded version
of factorizing 15 and random guess 7

input: a - random guess
       power -  value p 
'''
def c_amod15(a, power):
    U = QuantumCircuit(4)
    for iteration in range(power):
        U.swap(2, 3)
        U.swap(1, 2)
        U.swap(0, 1)
        for q in range(4):
            U.x(q)
        U.to_gate()
        U.name = '%i^%i mode 15' % (a, power)
        c_U = U.control()
        return c_U

# define 8-qubit
n_count = 8
# define our guess is 7
a = 7


'''
quantum circuit for QFT
input: n - number of deisgned qubit
'''
def qft_dagger(n):
    qc = QuantumCircuit(n)
    for qubit in range(n // 2):
        qc.swap(qubit, n - qubit - 1)
    for j in range(n):
        for m in range(j):
            qc.cp(-np.pi / float(2 ** (j - m)), m, j)
        qc.h(j)
    qc.name = 'QFT dagger'
    return qc


if __name__ == '__main__':

    # create test environment
    qc = QuantumCircuit(n_count + 4, n_count)

    # initalize quantum circuit - superposition all qubits
    for q in range(n_count):
        qc.h(q)
    
    qc.x(3 + n_count)
    
    # add modular exponentiation function
    for q in range(n_count):
        qc.append(c_amod15(a, 2 ** q), [q] + [i + n_count for i in range(4)])
    
    # add QFT function
    qc.append(qft_dagger(n_count), range(n_count))

    # perform quantum measurement
    qc.measure(range(n_count), range(n_count))
    
    # print out quantum circuit
    print(qc.draw('text'))

    # set up simulator
    backend = Aer.get_backend('qasm_simulator')
    results = execute(qc, backend, shots = 2048).result()
    counts = results.get_counts()
    plot_histogram(counts)