# https://github.com/austinjhunt/qiskit/blob/c8e5bbe46f4fa93ef8388efc87af0d5074873bdc/shors-algorithm/util.py
""" Utility methods used for Shor's algorithm in main.py """

import matplotlib.pyplot as plt
import numpy as np 
import math 
from qiskit import QuantumCircuit, Aer, transpile, assemble
from qiskit.visualization import plot_histogram
from math import gcd
from numpy.random import randint
import pandas as pd
from fractions import Fraction
import os 
base_dir = os.path.dirname(__file__) 

def a2jmodN(a, j, N):
    """Compute a^{2^j} (mod N) by repeated squaring. 
    This method is intended to provide some scalability to this algorithm 
    by removing the statically set N=15 requirement. This is not currently used
    as I am not yet sure where to leverage it. 
    """
    for i in range(j):
        a = np.mod(a**2, N)
    return a

def c_amod15(g, p):
    """  
    To create U^p where U is a unitary operator, we repeat the circuit p times. 
    This is specifically written to achieve controlled multiplication by g mod 15.
    Where: 
    g is some "bad" integer guess between 2 and 14 inclusive, 
    N is 15 (the number whose prime factors we want), and 
    p is such that g^p = m * N + 1 or g^p mod m*N = 1. 

    Tying this back into the README, if we consider our input superposition of all
    possible values of the power p, we are obtaining an output superposition of all 
    of the corresponding remainder values defined by r = g^p mod m*N, with N statically set as 15. 
    """

    # If g is 3, 5, 6, 9, 10, 12, or 14 raise an error. 
    if g not in [2,4,7,8,11,13]:
        raise ValueError("'a' must be 2,4,7,8,11 or 13")
    U = QuantumCircuit(4) 
    ## Repeat the following execution p times to achieve U^p        
    for i in range(p):   
        if g in [2,13]: 
            # use SWAP gate (equivalent to a state swap; classical logic gate)
            # to swap qubit 0 with qubit 1, qubit 1 with qubit 2, qubit 2 with qubit 3. 
            U.swap(0,1)
            U.swap(1,2)
            U.swap(2,3)
        if g in [7,8]:
            # use SWAP gate (equivalent to a state swap; classical logic gate)
            # to swap qubit 2 with qubit 3, qubit 1 with qubit 2, qubit 0 with qubit 1. 
            U.swap(2,3)
            U.swap(1,2)
            U.swap(0,1)
        if g in [4, 11]:
            # use SWAP gate (equivalent to a state swap; classical logic gate)
            # to swap qubit 1 with qubit 3, qubit 0 with qubit 2 
            U.swap(1,3)
            U.swap(0,2)
        if g in [7,11,13]:
            # apply Pauli-X gate on qubits 0, 1, 2, and 3
            # this gate essentially acts like a classical bit flip.
            # equivalent to a pi radian rotation about the X axis. 
            for q in range(4):
                U.x(q)
    # Create one gate out of this whole repeated circuit.
    U = U.to_gate()
    # Name it after what it is specifically doing. 
    U.name = "%i^%i mod 15" % (g, p)
    # Return a controlled version of the gate U with default 1 contro added to gate. 
    c_U = U.control()
    return c_U

def qft_dagger(n):
    """Apply N-qubit QFTdagger to the first n qubits of a circuit.
    
    QFT dagger is the conjugate transpose of the Quantum Fourier Transform,
    where the quantum fourier transform is used to find a frequency of a 
    given superposition. 
    """ 
    # Defining a new quantum circuit of n qubits. 
    circuit = QuantumCircuit(n)

    # for each qubit in the first half of that circuit
    for qubit in range(n//2):
        # Apply a SWAP gate to swap:
        #  qubit 0 with the n - 1 qubit
        #  qubit 1 with the n - 2 qubit
        #  qubit 2 with the n - 3 qubit
        # ... such that you are inverting the first n qubits
        circuit.swap(qubit, n-qubit-1)

    # for each qubit j in the full circuit
    for j in range(n):
        # for each qubit preceding j 
        for m in range(j): 
            # apply a Controlled-Phase gate
            # This is a diagonal and symmetric gate that induces a 
            # phase/rotation on the state of the target qubit, depending on the control state.
            # Define the rotation angle as (-pi) / 2^(j-m)
            rotation_angle = -math.pi/float(2**(j-m))
            # apply the controlled-phase gate to qubit j using that rotation angle, 
            # using m as the control qubit, and using j as the target qubit
            circuit.cp(rotation_angle, control_qubit=m, target_qubit=j)
        
        # Apply Hadamard gate to qubit j to 
        # put it into a superposition such that probability of 0 = probability of 1
        circuit.h(j)

    # Give the Quantum Fourier Transform Dagger (conjugate transpose) 
    # circuit a name and then return it 
    circuit.name = "QFT†"
    return circuit
 

 
def assert_g_not_a_trivial_factor_of_N(g, N):
    """ You need to ensure that the initial guess g is not 
    a direct, or trivial, factor of N because Shor's algorithm will not work if so.
    Euclid's algorithm for greatest common divisor can be used for this check.
    If the GCD is 1, then g is not a trivial factor of N.
    """
    print(f'Asserting that g={g} is not a trivial factor of N={N}')
    assert (gcd(g, N) == 1)

def qpe_amod15(g):
    """ 
    Apply quantum phase estimation to estimate the phase for some "bad" integer guess g
    and some large number N whose factors need to be determined. 
    We want to find the period p such that g^p = m * N + 1, or g^p mod (m * N) = 1.
    """
    # Here we define the number of "counting qubits" such that we can 'count' on the
    # first n_count qubits of our circuit. 
    n_count = 8 
    # Create a quantum circuit; when defining the circuit, 
    # we add an extra 4 qubits for the unitary operator U
    # to act on. This will have n_count + 4 quantum registers and just
    # n_count classical registers. The classical registers are to 
    # allow the mapping of quantum measurement results to classical bits.
    circuit = QuantumCircuit(4 + n_count, n_count)

    # for those first n_count counting qubits, we want to initialize each one 
    # to a superposition state using an H (Hadamard) gate.
    # an H gate (Hadamard gate) transforms the qubit such that
    # the probability of measuring 0 from the qubit is equal to the 
    # probability of measuring 1 from the qubit
    # Very common initialization step in quantum programming.
    for q in range(n_count):
        circuit.h(q) 

    # apply single-qubit Pauli-X gate to the last qubit.
    # the Pauli-X gate is equivalent to a classical bit flip, i.e. where 0 becomes 1 and
    # 1 becomes 0.
    circuit.x(3+n_count) 

    # For each of the n_count "counting qubits", do controlled U operations 
    for qubit in range(n_count): 
        # for qubit i, append to the circuit a controlled U gate 
        # representing U^(2^i) repeated g mod 15 circuits (with each repetition multiplying on itself)
        # we ultimately want to obtain a phase s / p where g^p mod N = 1. 
        # This allows us to represent a superposition of all remainders r where 
        # each r_i = (guess ^ 2^i) mod N, with N = 15.
        # We can then use that superposition of all remainders to find a period as discussed in the README.
        circuit.append(
            c_amod15(g, 2**qubit),  [qubit] + [i+n_count for i in range(4)])

    # We now have a superposition of all remainders r. 
    # We want to use the Quantum Fourier Transform to obtain a frequency from that superposition/wave function.
    # first, we need to append a QFTDagger gate (conjugate transpose of the QFT) to the circuit
    # to apply the inverse of the Quantum Fourier Transformation 
    # to the first n_count qubits of the circuit as defined above.
    circuit.append(qft_dagger(n_count), range(n_count)) # Do inverse-QFT

    # We want to measure the results from the first n_count qubits, and 
    # map those results into corresponding n_count classical bits of our quantum circuit.
    circuit.measure(range(n_count), range(n_count))

    # We use the AER simulator to simulate results. 
    aer_sim = Aer.get_backend('aer_simulator')
    # Setting memory=True below allows us to see a list of each sequential reading
    t_circuit = transpile(circuit, aer_sim)
    qobj = assemble(t_circuit, shots=1)
    # Obtain the result from the simulation and print / display those results.
    result = aer_sim.run(qobj, memory=True).result()
    readings = result.get_memory()
    print("Register Reading: " + readings[0])

    # Cast the register reading to an integer and divide by 2 to the power of n_count
    # to get the phase, which corresponds to the frequency obtained from the QFT as discussed
    # in the README, where the frequency is 1/p, and p is the value we want. 
    phase = int(readings[0],2)/(2**n_count)
    print("Corresponding Phase: %f" % phase)
    return phase
 