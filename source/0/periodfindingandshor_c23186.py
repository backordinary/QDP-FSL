# https://github.com/huwenqing0606/Quantum-Computing-using-Qiskit/blob/92bda1cff985fc4c529414433382739220d473b4/PeriodFindingAndShor.py
import numpy as np
from math import gcd
from numpy.random import randint
import pandas as pd
from fractions import Fraction
# importing Qiskit
from qiskit import IBMQ, Aer, transpile, assemble
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
# import basic plot tools
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram
# import inverse Quantum Fourier Transform
from QuantumFourierTransform import QFourier_inverse


# U |y> = |ay mod 15>
def c_amod15(a, power):
    """ Controlled multiplication by a mod 15 """
    """ Returns the controlled-U gate for a, repeated power times """
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


# Demonstrate the period-finding algorithm
def periodfinding_demonstration():
    # Specify variables
    n_count = 3  # number of counting qubits
    a = 7
    # Create QuantumCircuit with n_count counting qubits
    # plus 4 qubits for U to act on
    qc = QuantumCircuit(n_count + 4, n_count)
    # Initialize counting qubits
    # in state |+>
    for q in range(n_count):
        qc.h(q)
    qc.barrier()
    # And auxiliary register in state |1>
    qc.x(3 + n_count)
    qc.barrier()
    # Do controlled-U operations
    for q in range(n_count):
        qc.append(c_amod15(a, 2**q), [q] + [i+n_count for i in range(4)])
    qc.barrier()
    # Do inverse-QFT
    QFourier_inverse(qc, n_count)
    # Measure circuit
    qc.measure(range(n_count), range(n_count))
    qc.draw(fold=-1, output='mpl')  # -1 means 'do not fold' 
    plt.show()
    # simulate result
    aer_sim = Aer.get_backend('aer_simulator')
    t_qc = transpile(qc, aer_sim)
    qobj = assemble(t_qc)
    results = aer_sim.run(qobj).result()
    counts = results.get_counts()
    plot_histogram(counts)
    plt.show()
    # calculate the phases
    rows, measured_phases = [], []
    for output in counts:
        decimal = int(output, 2)  # Convert (base 2) string to decimal
        phase = decimal/(2**n_count)  # Find corresponding eigenvalue
        measured_phases.append(phase)
        # Add these values to the rows in our table:
        rows.append([f"{output}(bin) = {decimal:>3}(dec)", 
                    f"{decimal}/{2**n_count} = {phase:.2f}"])
    # Print the rows in a table
    headers=["Register Output", "Phase"]
    df = pd.DataFrame(rows, columns=headers)
    print(df)
    # phases and guesses for r
    rows = []
    for phase in measured_phases:
        frac = Fraction(phase).limit_denominator(15)
        rows.append([phase, f"{frac.numerator}/{frac.denominator}", frac.denominator])
    # Print as a table
    headers=["Phase", "Fraction", "Guess for r"]
    df = pd.DataFrame(rows, columns=headers)
    print(df)
    return None


# Demonstrate an example of Shor's factorization algorithm
def Shor_demonstration():
    # factoring N
    N = 15
    # The first step is to choose a random number a between 1 and N-1
    np.random.seed(1) # This is to make sure we get reproduceable results
    a = randint(2, 15)
    print(a)
    # check that the number a we picked is not a non-trivial factor of N
    gcd(a, N)
    # find the phase s/r for a mod 15
    phase = qpe_amod15(a) # Phase = s/r
    # estimate the period r
    Fraction(phase).limit_denominator(15) # Denominator should (hopefully!) tell us r
    frac = Fraction(phase).limit_denominator(15)
    s, r = frac.numerator, frac.denominator
    print(r)
    # factor
    guesses = [gcd(a**(r//2)-1, N), gcd(a**(r//2)+1, N)]
    print(guesses)
    # repeat until totally factored
    a = 7
    factor_found = False
    attempt = 0
    while not factor_found:
        attempt += 1
        print("\nAttempt %i:" % attempt)
        phase = qpe_amod15(a) # Phase = s/r
        frac = Fraction(phase).limit_denominator(N) # Denominator should (hopefully!) tell us r
        r = frac.denominator
        print("Result: r = %i" % r)
        if phase != 0:
            # Guesses for factors are gcd(x^{r/2} ±1 , 15)
            guesses = [gcd(a**(r//2)-1, N), gcd(a**(r//2)+1, N)]
            print("Guessed Factors: %i and %i" % (guesses[0], guesses[1]))
            for guess in guesses:
                if guess not in [1,N] and (N % guess) == 0: # Check to see if guess is a factor
                    print("*** Non-trivial factor found: %i ***" % guess)
                    factor_found = True
    return None


# periodic finding for N = 15
def qpe_amod15(a):
    n_count = 8
    qc = QuantumCircuit(4+n_count, n_count)
    for q in range(n_count):
        qc.h(q)     # Initialize counting qubits in state |+>
    qc.x(3+n_count) # And auxiliary register in state |1>
    for q in range(n_count): # Do controlled-U operations
        qc.append(c_amod15(a, 2**q), [q] + [i+n_count for i in range(4)])
    QFourier_inverse(qc, n_count)
    qc.measure(range(n_count), range(n_count))
    # Simulate Results
    aer_sim = Aer.get_backend('aer_simulator')
    # Setting memory=True below allows us to see a list of each sequential reading
    t_qc = transpile(qc, aer_sim)
    qobj = assemble(t_qc, shots=1)
    result = aer_sim.run(qobj, memory=True).result()
    readings = result.get_memory()
    print("Register Reading: " + readings[0])
    phase = int(readings[0],2)/(2**n_count)
    print("Corresponding Phase: %f" % phase)
    return phase


if __name__ == '__main__':
    demonstrate_periodfinding = 0
    if demonstrate_periodfinding:
        periodfinding_demonstration()
    else:
        Shor_demonstration()

