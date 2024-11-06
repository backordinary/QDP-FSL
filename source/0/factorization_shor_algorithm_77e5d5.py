# https://github.com/ezerc/Qiskit_Projects/blob/e4d5d3dfd5f029cdd3ccb0d88ac05b8f1684e9dd/Shor_Factorization/factorization_shor_algorithm.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 11:58:15 2021

@author: Ezequiel
"""


import sys
from qiskit import *
from qiskit.tools.visualization import plot_histogram
from qiskit.tools.monitor import job_monitor
import numpy as np
pi = np.pi
from fractions import Fraction
from math import gcd
from numpy.random import randint
from statistics import mode
from collections import Counter




###################### PRE-PROCESSING ######################

def perfect_power_check(x):
    ''' Check if a specific number is a perfect power of the form p^q, with p and q 
    integers. This is done by computing q-th roots and checking if they are integers.
    Note that p >=2 and q > 1. This is not the most efficient, but suffices for the 
    numbers considered in this program.
    
    Args:
        -x: Number to be checked
    '''
    
    # Maximum possible value of the exponent, if it were that p=2
    n = len(format(x,'b')) + 1
    
    found = False
    
    # Loop over potential values of the exponent and find roots
    for q in range(2,n):
        
        p = x**(1/q)
        
        # If p is integer we found the base
        if p - np.floor(p) == 0:
            found = True
            break
    
    # Return whether we x is a perfect power, if so also return base and exponent
    return found,int(p),int(q)

def pre_process(N):
    ''' Pre-processing steps before jumping into factorization with Shor's algorithm:
            -Check if N is equal to 0 or 1
            -Check if N is even
            -Check if N is a perfect power N = p**q
        
        Args:
            -N: Value we want to factor
    '''
    
    if N < 2:
        sys.exit('---> Dont need to factor this! Use a different value for N')
        
    
    if N%2 == 0:
        answer = input('This is even...would you like to try and factor {} instead?(y/n) '.format(int(N/2)))
        if answer == 'y':
            N = int(N/2)
            pre_process(N)
        else:
            sys.exit('---> No problem!')
            
    
    ppc,p,q = perfect_power_check(N)
    
    if ppc == True:
        sys.exit('---> {} is a perfect power of the form {}^{}, so there goes your factors'.format(N,p,q))
    
    return N

#######################################################

    
    
    
###################### QFT OPERATIONS ######################
    
def qft_rot(circuit,q_reg,nq,inverse):
    ''' First part of the Quantum Fourier Transform. This function performs rotations 
    on a specific target qubit, which are controlled by remaining qubits that have not 
    been rotated yet. This is therefore performed recursively in order to progressively 
    reduce the number of qubits affected by the operation. The particular order of operation 
    has been set such that it follows the notation order in Qiskit. The target qubit is 
    thus the last one in the circuit and the function recursively moves upwards.
      
    Args:
        -circuit: Circuit on which the gates act
        -q_reg: Specific quantum register on which the gates act
        -nq: Number of qubits in the circuit
        -inverse: If False, regular qft is computed. If True, inverse qft is computed
        '''
    # The parameter m sets the sign of the rotation angles
    if inverse:
        m = -1
    else:
        m = 1

    # If all qubits have been acted on return the circuit
    if nq == 0:        
        return circuit
    
    # Reduce by one the number of qubits in order to recursevely move through the circuit
    nq-=1
    
    # Act with a Hadamard gate on the target qubit
    circuit.h(q_reg[nq])
    
    # Controlled rotations on the target qubit, with each one of the remaining qubits as controls
    for q in range(nq):
    
        circuit.cu1(m*2*pi/2**(nq+1-q),q_reg[q],q_reg[nq])
    
    # Recursively call the function to cover the entire circuit
    return qft_rot(circuit,q_reg,nq,inverse)

def qft_swaps(circuit,q_reg,nq):
    '''Second part of the Quantum Fourier transform. This function swaps qubits 
    paired from top and bottom towards the middle.
      
    Args:
        -circuit: Circuit on which the gates act
        -q_reg: Specific quantum register on which the gates act
        -nq: Number of qubits in the circuit
    '''
       
    # Swapping of each pair
    for q in range(nq//2):       
        circuit.swap(q_reg[q],q_reg[nq-1-q])
        
        
def qft(circuit,q_reg,nq,inverse = False):
    '''Quantum Fourier Transform. The function is split into a series of controlled 
    rotations and swap operations. See Qiskit textbook section about QFT for details.
      
    Args:
        -circuit: Circuit on which the gates act
        -q_reg: Specific quantum register on which the gates act
        -nq: Number of qubits in the circuit
        -inverse: If False, regular qft is computed. If True, inverse qft is computed
    ''' 
    # Rotations on the qubits
    qft_rot(circuit,q_reg,nq,inverse)
    
    # Qubit swaps
    qft_swaps(circuit,q_reg,nq)
    
#############################################################################





###################### OPERATIONS FOR MODULAR EXPONENTIATION ######################
    
def egcd(a, b):
    ''' Computation of modular inverse using Euclid's algorithm.
    
    Args:
        -a: Value to invert and compute modulo of
        -b: Modulo value
    '''
    
    if a == 0:
        return (b, 0, 1)
    else:
        g, y, x = egcd(b % a, a)
        return (g, x - (b // a) * y, y)

def modinv(a, m):
    ''' Computation of modular inverse using Euclid's algorithm.
    
    Args:
        -a: Value to invert and compute modulo of
        -b: Modulo value
    '''
    
    g, x, y = egcd(a, m)
    if g != 1:
        
        sys.exit('---> Modular inverse does not exist for this seed value. Try another one.')

    else:
        return x % m
    
    
    
def ccrot(circuit,ctrl_A,ctrl_B,target,angle):
    ''' Doubly controlled rotations.
    
    Args:
        -circuit: Circuit control and target qubits
        -ctrl_A: First control qubit that determines if the operation is performed
        -ctrl_B: Second control qubit that determines if the operation is performed
        -target: Qubit on which the rotation is performed
        -angle: Rotation angle
    
    '''
    
    circuit.cu1(angle/2,ctrl_A,target)
    circuit.cx(ctrl_A,ctrl_B)
    circuit.cu1(-angle/2,ctrl_B,target)
    circuit.cx(ctrl_A,ctrl_B)
    circuit.cu1(angle/2,ctrl_B,target)
    
    
    
def fourier_adder(circuit,q_reg,nq,num):
    ''' Addition of a classical number in Fourier space. Action: |F(a)> ---> |F(a+b)>, 
    with |F(a)> the Fourier transform of |a> and b the classical number.
    
    Args:
        -circuit: Circuit on which the gates act
        -q_reg: Specific quantum register on which the gates act
        -nq: Number of qubits encoding the quantum number
        -num: Classical number to be added
    '''
    
    # Each rotation adds a binary component of num into the Fourier transformed state
    for q in range(nq):
        circuit.rz(2*pi*num/2**(nq-q),q_reg[q])
        
        
        
def c_fourier_adder(circuit,ctrl,q_reg,nq,num):
    ''' Controlled addition of a classical number in Fourier space. 
    Action: |F(a)>|c> ---> |F(a+b)>|c> if c=1, with |F(a)> the Fourier transform of 
    |a> and b the classical number.
    
    Args:
        -circuit: Circuit on which the gates act
        -ctrl: Control qubit that determines if the operation is performed
        -q_reg: Specific quantum register on which the gates act
        -nq: Number of qubits encoding the quantum number
        -num: Classical number to be added
    '''
    
    # Each rotation adds a binary component of num into the Fourier transformed state
    for q in range(nq):
        circuit.cu1(2*pi*num/2**(nq-q),ctrl,q_reg[q])

        
        

def cc_fourier_adder(circuit,ctrl_A,ctrl_B,q_reg,nq,num):
    ''' Doubly controlled addition of a classical number in Fourier space.
    Action: |F(a)>|c_a>|c_b> ---> |F(a+b)>|c_a>|c_b> if c_a=c_b=1, with |F(a)> the 
    Fourier transform of |a> and b the classical number.
    
    Args:
        -circuit: Circuit on which the gates act
        -ctrl_A: First control qubit that determines if the operation is performed
        -ctrl_B: Second control qubit that determines if the operation is performed
        -q_reg: Specific quantum register on which the gates act
        -nq: Number of qubits encoding the quantum number
        -num: Classical number to be added
    '''
    
    # Each rotation adds a binary component of num into the Fourier transformed state
    for q in range(nq):
        ccrot(circuit,ctrl_A,ctrl_B,q_reg[q],2*pi*num/2**(nq-q))
        



def cc_fourier_mod_adder(circuit,ctrl_A,ctrl_B,q_reg,nq,num,mod):
    ''' Doubly controlled modular addition of a classical number in Fourier space. 
    The function is conditional on the sum of the classical and quantum numbers to 
    be smaller than twice the modulo value. An ancilla is used to determine if the
    modulo value needs to be subtracted or not.
    Action: |F(a)>|c_a>|c_b> ---> |F[(a+b) mod N]>|c_a>|c_b> if c_a=c_b=1, with |F(a)> 
    the Fourier transform  of |a>, b the classical number and N the modulo value.
    
    Args:
        -circuit: Circuit on which the gates act
        -ctrl_A: First control qubit that determines if the operations is performed
        -ctrl_B: Second control qubit that determines if the operations is performed
        -q_reg: Specific quantum register on which the gates act
        -nq: Number of qubits encoding the quantum number
        -num: Classical number to be added
        -mod: Modulo value
    '''
   
    # Add the classical bit
    cc_fourier_adder(circuit,ctrl_A,ctrl_B,q_reg,nq,num)
    
    # Subtract the modulo value
    fourier_adder(circuit,q_reg,nq,-mod)
    
    # Flip the ancilla if the stored number is negative
    qft(circuit,q_reg,nq,inverse = True)
    circuit.cx(q_reg[nq-1],q_reg[nq])
    qft(circuit,q_reg,nq)
    
    # Add the modulo value if appropriate
    c_fourier_adder(circuit,q_reg[nq],q_reg,nq,mod)
    
    # Restore the value of the ancilla to |0> if appropriate
    cc_fourier_adder(circuit,ctrl_A,ctrl_B,q_reg,nq,-num)
    qft(circuit,q_reg,nq,inverse=True)
    circuit.x(q_reg[nq-1])
    circuit.cx(q_reg[nq-1],q_reg[nq])
    circuit.x(q_reg[nq-1])
    qft(circuit,q_reg,nq)
    cc_fourier_adder(circuit,ctrl_A,ctrl_B,q_reg,nq,num)
    


def c_modular_multiplier(circuit,ctrl,q_main,q_ops,nq,num,mod):
    ''' Controlled modular multiplication and addition of a classical number with two 
    quantum registers. Action: |x>|b>|c> ---> |x>|b+(ax)mod N >|c> if c=1, with |x>|b> 
    two quantum registers, a the classical number and N the modular value. By default, 
    two extra qubits are added on the operations register, one to prevent overflow in 
    the additions and a second one as modular addition requires an ancillary qubit.
    
    Args:
        -circuit: Circuit on which the gates act
        -ctrl: Control qubit that determines if the operation is performed
        -q_main: Main quantum register encoding the number to be multiplied
        -q_ops: Operations quantum register on which operations are performed
        -nq: Number of qubits in the main register. By definition, the operations
             register contains nq+2 qubits
        -num: Classical factor multiplying the number encoded in the main register
        -mod: Modulo value
    '''
    
    # Take the second register into Fourier space to perform modular additions
    qft(circuit,q_ops,nq+1)
    
    # Modular additions conditioned by qubits in the first register
    for q in range(nq):
        cc_fourier_mod_adder(circuit,ctrl,q_main[q],q_ops,nq+1,(2**q*num)%mod,mod)
    
    # Bring back the second register from Fourier space
    qft(circuit,q_ops,nq+1,inverse = True)

    

def c_modular_exp(circuit,ctrl,q_main,q_ops,nq,factor,mod):
    '''Controlled modular exponentiation. Action: |x>|0>|c> ---> |(ax)mod N>|0>|c> 
    if c=1, with a the factor to be exponentiated and N the modular value. The 
    operations (second) register is required to perform addition and multiplication
    operations on the side. By default, these require the operations register to
    contain two extra qubits with respect to the main register.
    
    Args:
        -circuit: Circuit on which the gates act
        -ctrl: Control qubit that determines if the operation is performed
        -q_main: Main quantum register encoding the number to be multiplied
        -q_ops: Operations quantum register on which operations are performed
        -nq: Number of qubits in the main register. By definition, the operations
             register contains nq+2 qubits
        -factor: Factor to be exponentiated, multiplying the number encoded in 
                 the main register
        -mod: Modulo value
    '''
    
    
    # Modular multiplication yielding the result in the second register
    #circuit.append(modular_multiplier(nq,factor,mod),circuit.qubits[:(2*nq+2)])
    c_modular_multiplier(circuit,ctrl,q_main,q_ops,nq,factor,mod)

    # Swap qubits from the second to the first register, excluding the extra qubit
    for qubit in range(nq):
        circuit.cswap(ctrl,q_main[qubit],q_ops[qubit])
        
    # Compute modular inverse of 'factor'
    inv_factor = modinv(factor,mod)
    
    # Restore value of the second register to |0>
    #circuit.append(modular_multiplier(nq,-inv_factor,mod),circuit.qubits[:(2*nq+2)])
    c_modular_multiplier(circuit,ctrl,q_main,q_ops,nq,-inv_factor,mod)
    
#############################################################################





###################### SHOR CIRCUIT AND POST-PROCESSING ######################
    
def shor_factoring(a,N):
    ''' Shor's algorithm. First obtain the period of a mod N, then use it
    to estimate a pair of factors of N. This implementation follows the 
    steps of Ref.[1], requiring a total of (2n+3) qubits, with one being 
    used for phase estimation by means of a sequential QFT.
    
        Args:
            -a: Seed value for period finding
            -N: Number to be factored
    '''
    
    # Size of the main register encoding the number to be factored
    nq = len(format(N,'b'))

    # We define a total of three quantum register: the main one, with size 
    # 'nq', encodes the state on which we perform modular exponentiation to 
    # extract the period; the operations one, which is required to perform 
    # the modular exponentiation operation, using two extra qubits; and an
    # ancilla, which is used to perform phase estimation and QFT using 
    # sequential measurements
    q_main = QuantumRegister(nq)
    q_ops = QuantumRegister(nq+2)
    q_meas = QuantumRegister(1)


    # Number of times we measure the system to extract the period
    n_meas = 2*nq

    # The classical register includes a main register on which we store every
    # measurement performed by the sequential QFT, and a second one which
    # stores locally each measurement to be used as a control in the next step
    # of the sequential QFT
    cr = ClassicalRegister(n_meas)
    signal = ClassicalRegister(1)

    circ = QuantumCircuit(q_meas,q_main,q_ops,cr,signal)

    # Initialization of the state |1> in the main register
    circ.x(q_main[1])
    circ.barrier()

    # Apply Sequential inverse QFT
    for q in range(n_meas):
    
        # Reset measurement ancilla to|0> if we previously measured |1>
        circ.x(0).c_if(signal,1)
    
        # Rotate measurement ancilla into a superposition state for pahse estimation
        circ.h(0)
    
        # Perform modular exponentiation controlled by the measurement ancilla
        c_modular_exp(circ,q_meas[0],q_main,q_ops,nq,a**2**(n_meas-1-q),N)
    
        # Rotations associated with QFT, performed according to previous measurments history
        for k in range(2**q):
            circ.rz(-2*pi*k/2**(q+1),0).c_if(cr,k)
    
        # Final rotation associated with QFT
        circ.h(0)
    
        # Store measurement history
        circ.measure(0,cr[q])
    
        # Store measurement to reset measurement ancilla
        circ.measure(0,signal[0])
    
    print(' ')
    print('Shor circuit execution started')
    sim = Aer.get_backend('qasm_simulator')
    result = execute(circ, backend = sim, shots = 2048).result()
    print('Execution finished')
    print(' ')
    
    
    # Extract counts from circuit execution
    raw_counts = result.get_counts()

    # Join outputs into a single binary number (they are extracted separated
    #due to having two classicla registers)
    counts = {}
    for key in raw_counts:
        new_key = key.split()
        new_key = ''.join(new_key)
        counts[new_key] = raw_counts[key]
          
    # Extract estimated phases and compute the periods using continuous fractions    
    period_values = []
    for bin_output in counts:
    
        # Convert binary output into decimal
        dec_output = int(bin_output, 2)

        # Find measured phase
        phase = dec_output/(2**n_meas) 
        # Express phase as a fraction
        frac = Fraction(phase).limit_denominator(N)

        # Use the denominator as a guess for the period if even
        if frac.denominator%2 ==0:
            period_values.append(frac.denominator)
            
            
    # If no even periods were found need to retry with different value of 'a'
    if period_values == []:
        
        answer = input('All period guesses were odd. Would you like try another seed value?(y/n) '.format(r))
        
        if answer == 'y':
            
            print('What value would you like to try instead? The previous one was a={} '.format(a))
            a_new = int(input())
            shor_factoring(a_new,N)
        
        else:
            sys.exit('---> No problem!')
            

    # Find number of counts per each period guess
    period_freqs = Counter(period_values)
    
    # Find best guess amoing the three most frequent ones
    period = 0
    for r,freq in period_freqs.most_common(3):
        
        # Make sure this is not a bad guess
        if gcd(a**(r//2)-1, N) != gcd(a**(r//2)+1, N):
            # Define period as the most frequent of good guesses
            period = r
            break
    
    if period==0:
        # If the three most frequent guesses were not good probably period is not even
        print('Whoops, only found factor 1, probably actual period is odd, try again with different seed?')
        
    else:
        
        print('The period guess is {}'.format(period))
        print(' ')
    
        # Extract factors from the period guess
        factors = [gcd(a**(period//2)-1, N), gcd(a**(period//2)+1, N)]
        print('Estimated factors of {}: {} and {}'.format(N,factors[0],factors[1]))
    
        # If true period was odd we will have output a bad guess of factors

        
    
##################################################################



    
###################### FACTORIZATION WRAPPER ######################
    
def factorize(N):
    ''' Factorization of a number. If the number is neither 0 or 1, nor even,
    nor a perfect power, this function uses Shor's algorithm to obtain factors
    of a desired number.
    
    Args:
        -N: Number we want to factor
    '''
    
    # Check if N is an easy case, i.e. if it is 0, 1, even or a perfect power
    # If even, N/2 might be considered for factoring if odd
    N = pre_process(N)

    # If not an easy case, use Shor's algorithm, ask for period finding seed 'a'
    print('Not a trivial case, Shor\'s algorithm will be used.')
    print('')
    print('Provide a seed value, between 2 and {} and such that gcd(N,value)=1, for period finding:'.format(N))
    a = int(input())
    shor_factoring(a,N)
    
##################################################################