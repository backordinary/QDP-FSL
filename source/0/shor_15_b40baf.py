# https://github.com/JoeLsvj/Shor-s-algorithm---Thesis/blob/603e41d4731bbdf4b41a786553bd4bc8b2080988/code%20-%20qiskit/shor_15.py
import qiskit.tools.jupyter
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, Aer, transpile, assemble, ClassicalRegister, QuantumRegister, execute
from qiskit.visualization import plot_histogram, circuit_drawer
from math import gcd
from numpy.random import randint
import pandas as pd
from fractions import Fraction
from qiskit import BasicAer
import qiskit

def plot_period(N,a):
    # Calculate the plotting data
    xvals = np.arange(N)
    yvals = [np.mod(a**x, N) for x in xvals]
    # Use matplotlib to display it nicely
    fig, ax = plt.subplots()
    ax.plot(xvals, yvals, linewidth=1, linestyle='dotted', marker='x')
    ax.set(xlabel='$x$', ylabel='$%i^x$ mod $%i$' % (a, N),
        title="Example of Periodic Function in Shor's Algorithm")
    try:  # plot r on the graph
        r = yvals[1:].index(1) + 1
        plt.annotate('', xy=(0, 1), xytext=(r, 1),
                    arrowprops=dict(arrowstyle='<->'))
        plt.annotate('$r=%i$' % r, xy=(r/3, 1.5))
    except ValueError:
        print('Could not find period, check a < N and have no common factors.')
    ax.set(xlabel='Number of applications of U', ylabel='End state of register',
        title="Effect of Successive Applications of U")
    fig

def c_amod15(a, power):
    """Controlled multiplication by a mod 15"""
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

plot_period(55,4)

# Specify variables
N = 15
n_count = int(np.ceil(np.log2(N))) # number of counting qubits
a = 7
def qft_dagger(n):
    """n-qubit QFTdagger the first n qubits in circ"""
    qc = QuantumCircuit(n)
    # Don't forget the Swaps!
    for qubit in range(n//2):
        qc.swap(qubit, n-qubit-1)
    for j in range(n):
        for m in range(j):
            qc.cp(-np.pi/float(2**(j-m)), m, j)
        qc.h(j)
    qc.name = "QFT†"
    return qc
# Create QuantumCircuit with n_count counting qubits
# plus 4 qubits for U to act on
qc = QuantumCircuit(3*n_count, 3*n_count)

# Initialize counting qubits
# in state |+>
for q in range(2*n_count):
    qc.h(q)
    
# And auxiliary register in state |1>
qc.x(3*n_count-1)

# Do controlled-U operations
for q in range(2*n_count):
    qc.append(c_amod15(a, 2**q), [q] + [i+2*n_count for i in range(n_count)])

#qc.measure(range(2*n_count,3*n_count), range(2*n_count,3*n_count))

# Do inverse-QFT
qc.append(qft_dagger(2*n_count), range(2*n_count))

# Measure circuit
qc.measure(range(2*n_count), range(2*n_count))
qc.draw('mpl', scale = 0.3, fold = -1)  # -1 means 'do not fold'

'''
aer_sim = Aer.get_backend('aer_simulator')
t_qc = transpile(qc, aer_sim)
results = aer_sim.run(t_qc).result()
counts = results.get_counts()
plot_histogram(counts)
'''

backend = BasicAer.get_backend('qasm_simulator')
job = execute(qc, backend, shots=1000)
result = job.result()
counts = result.get_counts(qc)
print(counts)
plot_histogram(data=counts, figsize=(14,10))

rows, measured_phases = [], []
for output in counts:
    decimal = int(output, 2)  # Convert (base 2) string to decimal
    phase = decimal/(2**n_count)  # Find corresponding eigenvalue
    measured_phases.append(phase)
    # Add these values to the rows in our table:
    rows.append([f"{output}(bin) = {decimal:>3}(dec)",
                 f"{decimal}/{2**n_count} = {phase:.2f}"])
# Print the rows in a table
headers = ["Register Output", "Phase"]
df = pd.DataFrame(rows, columns=headers)
print(df)
Fraction(0.666)
# Get fraction that most closely resembles 0.666
# with denominator < 15
Fraction(0.666).limit_denominator(15)
rows = []
for phase in measured_phases:
    frac = Fraction(phase).limit_denominator(15)
    rows.append(
        [phase, f"{frac.numerator}/{frac.denominator}", frac.denominator])
# Print as a table
headers = ["Phase", "Fraction", "Guess for r"]
df = pd.DataFrame(rows, columns=headers)
print(df)
def a2jmodN(a, j, N):
    """Compute a^{2^j} (mod N) by repeated squaring"""
    for i in range(j):
        a = np.mod(a**2, N)
    return a

N = 15
np.random.seed(1) # This is to make sure we get reproduceable results
a = randint(2, 15)
from math import gcd # greatest common divisor
gcd(a, N)
def qpe_amod15(a):
    n_count = 8
    qc = QuantumCircuit(4+n_count, n_count)
    for q in range(n_count):
        qc.h(q)     # Initialize counting qubits in state |+>
    qc.x(3+n_count) # And auxiliary register in state |1>
    for q in range(n_count): # Do controlled-U operations
        qc.append(c_amod15(a, 2**q), 
                 [q] + [i+n_count for i in range(4)])
    qc.append(qft_dagger(n_count), range(n_count)) # Do inverse-QFT
    qc.measure(range(n_count), range(n_count))
    qc.draw('mpl', scale=0.3, fold=-1)
    # Simulate Results
    aer_sim = Aer.get_backend('aer_simulator')
    # Setting memory=True below allows us to see a list of each sequential reading
    t_qc = transpile(qc, aer_sim)
    result = aer_sim.run(t_qc, shots=1, memory=True).result()
    readings = result.get_memory()
    print("Register Reading: " + readings[0])
    phase = int(readings[0],2)/(2**n_count)
    print("Corresponding Phase: %f" % phase)
    return phase


phase = qpe_amod15(a)  # Phase = s/r
# Denominator should (hopefully!) tell us r
Fraction(phase).limit_denominator(15)
frac = Fraction(phase).limit_denominator(15)
s, r = frac.numerator, frac.denominator
print(r)
guesses = [gcd(a**(r//2)-1, N), gcd(a**(r//2)+1, N)]
print(guesses)
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
