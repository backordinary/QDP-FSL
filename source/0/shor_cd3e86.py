# https://github.com/parasol4791/quantumComp/blob/0a9a56334d10280e86376df0d57fdf8f4051093d/algos/shor.py
# Peter Shor's algorithm for quantum factorization of integers
import random
from math import gcd
from fractions import Fraction
import pandas as pd
from qiskit import QuantumCircuit
from qft import qft_dagger
from utils.backends import get_job_aer


def c_amod15(a, power):
    """Controlled multiplication by a mod 15.
        It's a special case of doing an exponential controlled
        a^power mod 15"""
    if a not in [2,7,8,11,13]:
        raise ValueError("'a' must be 2,7,8,11 or 13")
    U = QuantumCircuit(4)
    for iteration in range(power):
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
    U.name = "%i^%i mod 15" % (a, power)
    c_U = U.control()
    return c_U

def runShorCircuit(a, N, n_count):
    """Create and run Shor's quantum cirtuit.
        Return result counts"""
    # Create QuantumCircuit with n_count counting qubits
    # plus 4 qubits for U to act on
    qc = QuantumCircuit(n_count + 4, n_count)

    # Initialize counting qubits
    # in state |+>
    for q in range(n_count):
        qc.h(q)

    # And auxiliary register in state |1>
    qc.x(3 + n_count)

    # Do controlled-U operations
    for q in range(n_count):
        qc.append(c_amod15(a, 2 ** q),
                  [q] + [i + n_count for i in range(4)])

    # Do inverse-QFT
    qc.append(qft_dagger(n_count), range(n_count))

    # Measure circuit
    qc.measure(range(n_count), range(n_count))
    qc.draw(fold=-1)  # -1 means 'do not fold'

    # Simulation
    job = get_job_aer(qc, memory=True)
    return job.result().get_counts()


def output_to_period(output, n_count, N):
    """Converts measured binary output of Shor's circuit to a period r.
        n_count is a number of counting qubits.
        N is the number we are factoring"""
    decimal = int(output, 2)
    phase = decimal / (2**n_count)
    frac = Fraction(phase).limit_denominator(N)  # limit fraction denominators to range [1..N]
    return decimal, phase, frac, frac.denominator

def printPeriodicity(counts, n_count, N):
    """Prints periodicities of the phase"""
    rows = []
    for output in counts.keys():
        decimal, phase, frac, r = output_to_period(output, n_count, N)
        rows.append([f"{output}(bin) = {decimal:>3}(dec)",
                     f"{decimal}/{2 ** n_count} = {phase:.2f}",
                     f"{frac.numerator}/{r}", r])
    # Print the rows in a table
    headers = ["Register Output", "Phase", "Fraction", "Guess for r"]
    df = pd.DataFrame(rows, columns=headers)
    print(df)

def getPeriod(counts, n_count, N):
    """Returns period r from measured result counts"""
    periods = {}
    for output in counts.keys():
        _, _, _, r = output_to_period(output, n_count, N)
        if r in periods.keys():
            periods[r] += 1
        else:
            periods[r] = 1
    pList = [(ct, p) for p, ct in periods.items()]
    pList.sort(reverse=True)
    if len(pList) == 1:
        return pList[0][1]
    # Same top counts - choose non-trivial period
    elif pList[0][0] == pList[1][0]:
        return pList[0][1] if pList[0][1] != 1 else pList[1][1]
    # Return period with the top count
    else:
        return pList[0][1]


if __name__ == "__main__":
    # Finding periodicity r in function a^r mod N = 1
    # Phase is expressed as s/r, where s is a random number in range [0..r-1], r - periodicity

    # Specify variables
    n_count = 8  # number of counting qubits
    N = 15  # a number to factorize
    aList = [2,7,8,11,13]

    factorFound = False
    attempts = 1
    while not factorFound:
        print(f"\nAttempt {attempts}")
        attempts += 1
        a = aList[random.randint(0,4)]
        print(f"a = {a}")
        counts = runShorCircuit(a, N, n_count)
        print(f"Simulation counts: {counts}")
        # Simulation counts: {'01000000': 255, '11000000': 266, '00000000': 255, '10000000': 248}

        printPeriodicity(counts, n_count, N)
        # Note, it yields correct value of r only 50% of the time
        # The rest are either s = 0, or s and r are not co-prime numbers.
        #             Register Output           Phase Fraction  Guess for r
        # 0  11000000(bin) = 192(dec)  192/256 = 0.75      3/4            4
        # 1  10000000(bin) = 128(dec)  128/256 = 0.50      1/2            2
        # 2  00000000(bin) =   0(dec)    0/256 = 0.00      0/1            1
        # 3  01000000(bin) =  64(dec)   64/256 = 0.25      1/4            4

        r = getPeriod(counts, n_count, N)
        print(f"Period = {r}")
        if r % 2 != 0:
            continue

        guesses = [gcd(a**(r//2) - 1, N), gcd(a**(r//2) + 1, N)]
        print(f"Guesses: {guesses}")
        for g in guesses:
            if g not in [1, N] and N % g == 0:
                print(f"Found factor = {g}")
                factorFound = True

