# https://github.com/AriGood/Factor_Machine/blob/948c8446f7ba41dc05caa0fb6566cea2128460e9/Shor_Simulator.py
import math
from random import randrange
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer
import time
import sys

Semiprime = int(sys.argv[1])


def shor_algorithm(n, num_trials):
    factors = {}
    for i in range(num_trials):
        # Choose a random number `a` that is relatively prime to `n`
        a = randrange(2, n)
        while math.gcd(a, n) != 1:
            a = randrange(2, n)

        # Initialize qubits
        q = QuantumRegister(math.ceil(math.log2(n)) + 1)
        c = ClassicalRegister(math.ceil(math.log2(n)) + 1)
        qc = QuantumCircuit(q, c)

        # Apply the quantum Fourier transform
        qc.h(q[0])
        for i in range(1, len(q)):
            qc.h(q[i])
            qc.cp(2*math.pi/2**(i+1), q[i-1], q[i])
        qc.barrier()

        # Apply the modular exponentiation
        qc.x(q[0])
        qc.cp(2*math.pi/a, q[0], q[1])
        for i in range(1, len(q)-1):
            qc.cp(2*math.pi/a, q[i], q[i+1])
        qc.barrier()

        # Apply the inverse quantum Fourier transform
        for i in range(len(q)-1, 0, -1):
            qc.cp(-2*math.pi/2**(i+1), q[i-1], q[i])
            qc.h(q[i])
        qc.h(q[0])
        qc.barrier()

        # Measure the qubits
        for i in range(len(q)):
            qc.measure(q[i], c[i])
        backend = Aer.get_backend('qasm_simulator')
        job = execute(qc, backend, shots=1)
        result = job.result()
        k = int(result.get_counts().popitem()[0], 2)
        r = math.gcd(a**k - 1, n)
        if r != 1 and r != n:
            if r in factors:
                factors[r] += 1
            else:
                factors[r] = 1

    # return the factors that were found most often
    most_common_factor = max(factors, key=factors.get)
    return most_common_factor

start_time = time.time()

factors=shor_algorithm(Semiprime,10)

print(factors)

print((time.time() - start_time))

sys.stdout.flush()

