# https://github.com/xaviermccaig/Quantum-Thesis/blob/dbfb8fc762db2a34230152e7b1b603bcd898ffd0/Qiskit_factoring_15.py
#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing standard Qiskit libraries and configuring account
from qiskit import QuantumCircuit, execute, Aer, IBMQ
from qiskit.compiler import transpile, assemble
from qiskit.tools.jupyter import *
from qiskit.tools import *
from qiskit.visualization import *

# Importing general libraries
import matplotlib.pyplot as plt
import numpy as np
from math import gcd
from numpy.random import randint
import pandas as pd
from fractions import Fraction
get_ipython().run_line_magic('matplotlib', 'inline')

# Loading IBM Q account(s)
provider = IBMQ.load_account()


# In[2]:


###########################################################
# Performs the inverse Quantum Fourier Transform
# Purpose built for N=15 by the Qiskit team.
###########################################################
def qft_dagger(n):
    qc = QuantumCircuit(n)
    for qubit in range(n//2):
        qc.swap(qubit, n-qubit-1)
    for j in range(n):
        for m in range(j):
            qc.cp(-np.pi/float(2**(j-m)), m, j)       # Controlled phase gate which induces a phase given a control=1
        qc.h(j)
    qc.name = "QFT†"
    return qc
###########################################################
# Performs controlled unitary operation.
# Purpose-built for factoring 15 by the Qiskit team.
###########################################################
def c_amod15(a, power):
    if a not in [2,7,8,11,13]:
        raise ValueError("'a' must be 2,7,8,11 or 13") # Or else it would share factors with N=15.
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
###########################################################
# Performs the Quantum Phase Estimation routine, i.e.
# the order-finding routine of Shor's
###########################################################
def qpe_amod15(a):
    n_count = 8                                          # 8 in this case being the number of counting qubits
    qc = QuantumCircuit(4+n_count, n_count)
    for q in range(n_count):
        qc.h(q)                                          # Place counting qubits into superposition
        
    for q in range(n_count):                             # Apply controlled unitaries in doubling powers of 2
        qc.append(c_amod15(a, 2**q), 
                 [q] + [i+n_count for i in range(4)])
        
    qc.append(qft_dagger(n_count), range(n_count))       # Apply inverse Quantum Fourier Transform
    qc.measure(range(n_count), range(n_count))           # Measure the system
    qc.draw(output='mpl', filename='fac15.png',scale=0.35,fold=-1)
    backend = Aer.get_backend('qasm_simulator')
    job = execute(qc, backend=backend, memory=True, 
                shots=1000, optimization_level=3, )  # Creates a job object to run circuit on selected backend
                                                     # shots = number of times run, optim. = backend optimization
    job_monitor(job, interval = 2)                   # Initializes the job monitor to determine when job runs
    results = job.result()                           # Creates a results object from the run job                
    answer = results.get_memory()                    # Creates a 'counts' object which contains the results of the job                                                              
    
    print("Register Reading: " + answer[0])          # Reads the register
    phase = int(answer[0],2)/(2**n_count)            # Calculates the phase 
    print("Corresponding Phase: %f" % phase) 
    return phase
###########################################################
# Performs factoring operations until 
# nontrivial factors are found
###########################################################
def factorLoop(a, N):
    factor_found = False
    attempt = 0
    while not factor_found:
        attempt += 1
        print("\nAttempt %i:" % attempt)
        phase = qpe_amod15(a)                                         # Phase = s/r
        frac = Fraction(phase).limit_denominator(N)                   # Denominator should (hopefully!) tell us r
        r = frac.denominator
        print("Result: r = %i" % r)
        if phase != 0:
            guesses = [gcd(a**(r//2)-1, N), gcd(a**(r//2)+1, N)]      # Guesses for factors are gcd(x^{r/2} ±1 , 15)
            print("Guessed Factors: %i and %i" % (guesses[0], guesses[1]))
            for guess in guesses:
                if guess != 1 and guess != N and (N % guess) == 0:    # Check to see if guess is a factor
                    print("*** Non-trivial factor found: %i ***" % guess)
                    factor_found = True
###########################################################
# Main section
###########################################################
sim = True                           # Boolean for whether to run the circuit on simulator or hardware
if sim is True:
    backend = Aer.get_backend('qasm_simulator') 
    print("Running on qasm_simulator.\n")
else:
    backend = provider.backend.ibmq_16_melbourne
    print(f'Running on {backend}\n')

N = 15
a = 7                                 # This example designed for a = 7, but works for other values.
print(f'A is: {a}')
gcdA = gcd(a, N)

phase = qpe_amod15(a)                 # Phase = s/r
Fraction(phase).limit_denominator(N)  # Denominator should (hopefully!) tell us 'r' via continued fractions algorithm
frac = Fraction(phase).limit_denominator(N) 
s, r = frac.numerator, frac.denominator
print(f'Suggested order r: {r}')

guesses = [gcd(a**(r//2)-1, N), gcd(a**(r//2)+1, N)]
print(guesses)

factorLoop(a, N)                      # Initiate main loop for QPE and QFT inverse


# In[ ]:





# In[ ]:




