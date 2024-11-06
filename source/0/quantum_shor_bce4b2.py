# https://github.com/maripper/qubit_operators/blob/06dcff420d490f31a7896e0cf37713c33fa41263/base/utils/quantum_shor.py
from django.shortcuts import render

from qiskit.visualization import plot_bloch_multivector, plot_histogram

import requests as req

from django.http import JsonResponse

import random

import json

from django.views.decorators.csrf import csrf_exempt

from numpy import gcd
from numpy.random import seed, randint
# imports for Shor
from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram
from qiskit.circuit.library import QFT

from qiskit import IBMQ
from qiskit.aqua import QuantumInstance

def rsa(P, Q):
    N = P * Q # the hard number to crack!

    if N % 2 == 0:
        val = P if P % 2 == 0 else Q
        raise ValueError(f"{N} can not be divisible by 2.",
                         f"{P} and {Q} are incompatible with Shor's Algorithm.")

    L = (Q - 1) * (P - 1) # number of non-common factors (1, N)

    for E in range(2, L): # between [2, L)
        if gcd(L, E) * gcd(N, E) == 1: # coprime with both L and N
            break # E is public value

    D = 1
    while True:
        if D * E % L == 1 and D != E and D != N:
            break # D is private value
        D += 1

    return ((E, N), (D, N))

def dec(code, key):
    D, N = key
    return "".join([chr(((d**D) % N) + ord('A'))
                    for d in [int(d) for d in str(code)]])

def initialize_qubits(qc, n, m):
    qc.h(range(n)) # apply hadamard gates
    qc.x(n+m-1) # set qubit to 1
    
def a_x_mod15(a, x):
    if a not in [2,7,8,11,13]:
        raise ValueError("'a' must be 2,7,8,11 or 13")
    U = QuantumCircuit(4)        
    for iteration in range(x):
        if a in [2,13]:
            U.swap(0,1)
            U.swap(1,2)
            U.swap(2,3)
        if a in [7,8]:
            U.swap(2,3)
            U.swap(1,2)
            U.swap(0,1)
        if a == 11:
            U.swap(1,3)
            U.swap(0,2)
        if a in [7,11,13]:
            for q in range(4):
                U.x(q)
    U = U.to_gate()
    U.name = f"U({x})"
    c_U = U.control()
    return c_U

def modular_exponentiation(qc, n, m, a):
    for x in range(n):
        exponent = 2**x
        qc.append(a_x_mod15(a, exponent), 
                     [x] + list(range(n, n+m)))


def apply_iqft(qc, measurement_qubits):
    qc.append(QFT(len(measurement_qubits),
                             do_swaps=False).inverse(),
                         measurement_qubits)

def measure(qc, n):
    qc.measure(n, n)


def period_finder(n, m, a):
    
    # set up quantum circuit
    qc = QuantumCircuit(n+m, n)
    
    # initialize the qubits
    initialize_qubits(qc, n, m)
    qc.barrier()

    # apply modular exponentiation
    modular_exponentiation(qc, n, m, a)
    qc.barrier()

    # apply inverse QFT
    apply_iqft(qc, range(n))
    qc.barrier()

    # measure the n measurement qubits
    measure(qc, range(n))
    
    return qc


def rsa_page():
    N=15
    import random

    a = randint(2, N) # 1 < a < 
    el = [2,7,8,11,13]
    a = random.choice(el)

    if gcd(a, N) == 1: # a shares no factors
        print(f"{1} < {a} < {N}, {1 < a < N}")
    else: # a shares a factor
        P = gcd(a, N)
        Q = N // gcd(a, N)
        print(f"P = {P}\nQ = {Q}\n\n",
            f"{P} x {Q} = {N}, {P * Q == N}\n")

    n = 4; m = 4

    qc = period_finder(n, m, a)
    # qc.draw(output='mpl')

    simulator = Aer.get_backend('qasm_simulator')
    counts = execute(qc, backend=simulator).result().get_counts(qc)

    # plot_histogram(counts)
    print(counts)
    counts_dec = sorted([int(measured_value[::-1], 2)
                     for measured_value in counts])
    print(counts_dec)

    print("Measured periods:", end='\t')
    for measured_value in counts_dec:
        print(measured_value, end='\t')
        
    print(measured_value)
        #     sort through the possible exponents, finding those which satisfy two constraints:

        # the exponent, x, must be even and
        # a^(x/2) + 1 ≠ 0 (mod N)
# convert and add binary periods to list
    counts_dec = sorted([int(measured_value[::-1], 2)
                        for measured_value in counts])
    print(counts_dec)

    print("Measured periods:", end='\t')
    for measured_value in counts_dec:
        print(measured_value, end='\t')
    print(measured_value)
    # with period, x, you can find nontrivial factors, P and Q , of N with gcd(a^(x/2) ± 1, N) .
    factors = set()

    for x in counts_dec:
        guesses = [gcd(int((a ** (x/2))) + 1, N),
                gcd(int((a ** (x/2))) - 1, N)]
        for guess in guesses:
            # ignore trivial factors
            if guess != 1 and guess != N and N % guess == 0:
                factors.add(guess)

    if len(factors):
        P = factors.pop()
        Q = factors.pop() if len(factors) else N // P
        print(f"P = {P}\nQ = {Q}\n\n",
            f"{P} x {Q} = {N}, {P * Q == N}")
    else:
        P, Q = rsa_page()
    return P, Q
