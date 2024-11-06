# https://github.com/Aarun2/Quantum_Repo/blob/e54ff4be1615e5c1a5282857065008d3f5e34f73/Qiskit_Tutorials/Shor_Algo.py
# finding factors for large numbers. RSA
# larger the number more computations needed to find 
# prime factor. so more secure
# finding prime factor is a large time
# need 1000's years
# quantum computers can shorten this

from qiskit.aqua.algorithms import Shor
from qiskit.aqua import QuantumInstance
import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.tools.visualization import plot_histogram

backend = Aer.get_backend('qasm_simulator')
quantum_instance = QuantumInstance(backend, shots=1000)
my_shor = Shor(N=15, a=2, quantum_instance = quantum_instance)
Shor.run(my_shor)

# part1: Modular Exponentiation Function (factoring to period problem)
# part2: Quantum Fourier Transform (Quantum Speedup)
# part3: (computer factors of original problem)
# p = (a^r/2) - 1
# q = (a^r/2) + 1
# a = guess number
# r = period of mod exp function
# n = number trying to factor

def c_amod15(a, power):
    U = QuantumCircuit(4)
    for iteration in range(power):
        U.swap(2,3)
        U.swap(1,2)
        U.swap(0,1)
        for q in range(4):
            U.x(q)
    U=U.to_gate()
    U.name="%i^%i mod 15" %(a,power)
    c_U = U.control()
    return c_U

n_count = 8
a = 7

def qft_dagger(n):
    qc = QuantumCircuit(n)
    for qubit in range(n//2):
        qc.swap(qubit, n-qubit-1)
    for j in range(n):
        for m in range (j):
            qc.cp(-np.pi/float(2**(j-m)),m,j)
        qc.h(j)
    qc.name="QFT dagger"
    return qc

qc = QuantumCircuit(n_count + 4, n_count)

for q in range(n_count):
    qc.h(q)

qc.x(3+n_count)

for q in range(n_count):
    qc.append(c_amod15(a, 2**q), [q]+[i+n_count for i in range(4)])

qc.append(qft_dagger(n_count), range(n_count))

qc.measure(range(n_count), range(n_count))
qc.draw('text')

backend = Aer.get_backend('qasm_simulator')
results = execute(qc, backend, shots=2048).result()
counts = results.get_counts()
plot_histogram(counts)

# gives us three guesses for r: 1, 2, 4 from denominator of 4 phases (/256)
# lets get p and q
# they have co factors with n which we can find effeciently
# with r = 2, and a = 7 as seen before
# p = 48 and q = 50 which has factors 3 and 5
