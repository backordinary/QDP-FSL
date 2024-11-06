# https://github.com/kgeorgopoulos2/programbench/blob/d083b3e7f500dc599b53effdf02c5cc9cf9070fd/ctqwalk.py
# importing QISKit
import numpy as np
import scipy

from qiskit import Aer, IBMQ
from qiskit.quantum_info.operators import Operator
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, execute
from qiskit.extensions import HamiltonianGate

def getQState(qwalk, n):
    '''Method to get the quantum state before the measurement.'''
    sv_simulator = Aer.get_backend('statevector_simulator')

    # Execute and get counts
    result = execute(qwalk, sv_simulator).result()
    statevector = result.get_statevector(qwalk)

    # statevector

    b = []
    for k in range(0,len(statevector)):
        b.append(k)

    i = iter(statevector)
    j = iter(b)
    dct = dict(zip(j, i))

    # Create the dictionary without all the    
    qstate = {}
    for i in range(0,len(dct)):
        qstate[format(i, '0'+str(n)+'b')] = dct[i]
                                    
    return qstate

def getNonZeros(dct):
    '''Method that returns the positions of the state space that have non-zero amplitudes in
    the form of a dictionary.'''
    non_zeros = dict()

    for key in dct.keys():
        if (dct[key] != (0+0j)):
            non_zeros[key] = dct[key]
            
    return non_zeros

def getProbabilities(dct, n):
    '''Methods that calculates the probabilities of each position of the state space from
    the amplitudes of the quantum state. Returns a dictionary that contains the states with
    removed the ancilla and coin qubits.'''
    dct = getNonZeros(dct)
    probs = dict()
    
    probs = {k[-n:]:0 for k,v in dct.items()}
    for k,v in dct.items():
        probs[k[-n:]]+=np.abs(v)**2
    
    dec = [0]*len(probs.keys())
    probs_dec = dict()
    
    for key in probs.keys():
        probs_dec[int(key, 2)] = probs[key]
    
    return probs_dec

def getPosProb(n, t, meas):
    '''Method that returns the positions of the particle for a single number of coin flips,
    as well as the probabilities for the walker to be at each of the states.'''
    
    qwalk = ctqw(n, t, meas) # Return the CTQW circuit
    qstate = getQState(qwalk, n) # Return the statevector (i.e quantum state before measurement)
    dct = getProbabilities(qstate, n)
    
    pos = list(dct.keys())
    prob = list(dct.values())
    
    return pos,prob

def unitaryHam(N, t):
    '''Returns the operator resulting from the Hamiltonian exponentiation.'''
    # Define the Hamiltonian
    H = np.array([
        [0.+0.j]*N
    ]*N)

    H[0][N-1] = 1/2
    H[N-1][0] = 1/2
    for i in range(0, N):
        for j in range(0, N):
            if (j == i+1) or (j == i-1):
                H[i][j] = 1/2

    H = Operator(H)
    
    expHam = HamiltonianGate(H, t, label='CTQW') # This format includes time, t

    return expHam

# --------- Previous version -------------
# def unitaryHam(N, t):
#     # ATTENTION: this representation containes the experssion -iHt, i.e the power of the expression e^{-iHt}, and H=A*1/d => (0.-1.j) = -i
#     '''Returns the operator resulting from the Hamiltonian exponentiation.'''
#     miHt = np.array([
#         [0.+0.j]*N
#     ]*N)

#     miHt[0][N-1] = (0.-1.j)*t/2
#     miHt[N-1][0] = (0.-1.j)*t/2
#     for i in range(0, N):
#         for j in range(0, N):
#             if (j == i+1) or (j == i-1):
#                 miHt[i][j] = (0.-1.j)*t/2

#     # Perform the exponent e^{-iHt}
#     exp_miHt = scipy.linalg.expm(miHt)

#     U_c = Operator(exp_miHt)

#     return U_c
# -----------------------------------------

def ctqw(n, t, meas):
    '''Creates the continues time quantum walk circuit on an N-cycle'''
    N = 2**n
    U_c = unitaryHam(N, t)
    
    q = QuantumRegister(n, 'q')
    c = ClassicalRegister(n, 'c')
    circ = QuantumCircuit(q, c)

    circ.append(U_c, [q[i] for i in range(0, n)])
    
    if (meas == True):
        circ.measure(q,c)
    
    return circ