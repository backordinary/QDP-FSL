# https://github.com/physicalattraction/quantum-computing-tutorial/blob/051f72779490ccda6ba5a7a801a56b7ba8ae3e87/src/qiskit_3_7_shor_algorithm.py
"""
https://qiskit.org/textbook/ch-algorithms/shor.html
"""

from fractions import Fraction

import numpy as np
import pandas as pd
from math import pi, gcd
from qiskit import QuantumCircuit
from qiskit.circuit import ControlledGate

from utils import draw_quantum_circuit, print_histogram_from_real


def qft_dagger(n):
    """n-qubit QFTdagger the first n qubits in circ"""
    qc = QuantumCircuit(n)
    # Don't forget the Swaps!
    for qubit in range(n // 2):
        qc.swap(qubit, n - qubit - 1)
    for j in range(n):
        for m in range(j):
            qc.cp(-pi / float(2 ** (j - m)), m, j)
        qc.h(j)
    qc.name = "QFT†"
    return qc


def c_amod15(a, power) -> ControlledGate:
    """Controlled multiplication by a mod 15"""
    if a not in [2, 4, 7, 8, 11, 13]:
        raise ValueError("'a' must be 2, 4, 7, 8, 11 or 13")
    U = QuantumCircuit(4)
    for iteration in range(power):
        if a in [2, 13]:
            U.swap(0, 1)
            U.swap(1, 2)
            U.swap(2, 3)
        if a in [7, 8]:
            U.swap(2, 3)
            U.swap(1, 2)
            U.swap(0, 1)
        if a in [4, 11]:
            U.swap(1, 3)
            U.swap(0, 2)
        if a in [7, 11, 13]:
            for q in range(4):
                U.x(q)
    U = U.to_gate()
    U.name = f'{a}^{power} mod 15'
    c_U = U.control()
    return c_U


def qpe_amod15(a: int, use_real_device: bool = False):
    n_count = 8
    qc = QuantumCircuit(4 + n_count, n_count, name=f'qpe_{a}mod15')
    for q in range(n_count):
        qc.h(q)  # Initialize counting qubits in state |+>
    qc.x(3 + n_count)  # And auxiliary register in state |1>
    for q in range(n_count):  # Do controlled-U operations
        qc.append(c_amod15(a, 2 ** q),
                  [q] + [i + n_count for i in range(4)])
    qc.append(qft_dagger(n_count), range(n_count))  # Do inverse-QFT
    qc.measure(range(n_count), range(n_count))
    draw_quantum_circuit(qc)

    # Simulate Results
    if use_real_device:
        counts = print_histogram_from_real(qc)
    else:
        counts = draw_quantum_circuit(qc, draw_circuit=False, draw_simulate=True)
    measured_phases = []
    for output in counts:
        decimal = int(output, 2)  # Convert (base 2) string to decimal
        phase = decimal / (2 ** n_count)  # Find corresponding eigenvalue
        measured_phases.append(phase)

    phase = min([p for p in measured_phases if p != 0])
    print(f'Measured Phase for {a=}: {phase}')

    return phase


def factorize(nr: int, use_real_device: bool = False):
    """
    Find the prime factors of 15

    - Choose a co-prime a, where a∈[2,N−1] and the greatest common divisor of a and N is 1
         a = [2, 4, 7, 8, 11, 13, 14]
    - Find the order of a modulo N, i.e. the smallest integer r such that a^r mod N = 1
         for a = 7, r = 4:
         - a^2 mod 15 = 49 mod 15 = 4
         - a^3 mod 15 = 343 mod 15 = 13
         - a^4 mod 15 = 2401 mod 15 = 1
    - Find the factor of N by computing the greatest common divisor of a^(r/2) ± 1 and N
         gcd(7^2 + 1, 15) = gcd(50, 15) = 5
         gcd(7^2 - 1, 15) = gcd(48, 15) = 3

    In this procedure, the second step, finding the order of a modulo N,
    is the only quantum part, quantum order-finding.
    """

    if nr != 15:
        raise ValueError('This function has N=15 hardcoded in its circuit')
    guesses = {nr, 1}
    while len(guesses) <= 2:
        a = np.random.randint(2, nr)
        try:
            phase = qpe_amod15(a, use_real_device)  # Phase = s/r
        except ValueError:
            # Wrong value for a chosen. Try again
            return factorize(nr)
        frac = Fraction(phase).limit_denominator(nr)  # Denominator should (hopefully!) tell us r
        s, r = frac.numerator, frac.denominator

        guesses.add(gcd(a ** (r // 2) - 1, nr))
        guesses.add(gcd(a ** (r // 2) + 1, nr))
        for guess in guesses:
            if guess not in {nr, 1}:
                if nr % guess == 0:
                    other_guess = nr // guess
                    print(f'My guess is: {guess} * {other_guess}')
                else:
                    print(f'This guess was wrong: {guess}')
        if guesses == {nr, 1}:
            print('No new guesses found')


if __name__ == '__main__':
    factorize(15)
