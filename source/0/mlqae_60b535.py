# https://github.com/yukwangmin/IBMQ/blob/43cc6957c992eb7b812fa4ce950cfdbdebdd853f/AE/mlqae.py

import numpy as np
from scipy.optimize import brute

from qiskit import execute

from circuit import *
from utils import *


    
########################################################################################################################
## Circuit Start
########################################################################################################################

# n: number of qubits (domain)
# m: MLQLE circuits (m+1 circuits)
# b: Domain Volume
#
def makeMLQAECircuits(n: int, m: int, b:float, barrier=False):



    # construct first part of circuit
    q = QuantumRegister(n+1, 'q') # domain size + readout qubit
    c = ClassicalRegister(1, name='c0')
    
    qc_0 = QuantumCircuit(q,c)
    #if n>2:
    #    qc_0 = QuantumCircuit(q,a,c)

    # A build
    addA(qc_0, q, n, b, barrier)
    qc_0.barrier(q)


    evaluation_schedule = [0] + [2**j for j in range(m)]
    circuits = []
    for k in evaluation_schedule:
        qc_k = qc_0.copy()

        if k != 0: # Q power
            for i in range(k):
                addQ(qc_k, q, n, b, barrier)

        if barrier:
            qc_k.barrier(q)
                
        # measurement
        qc_k.measure(q[0], c[0])

        circuits += [qc_k]

    return circuits

########################################################################################################################
## Circuit End
########################################################################################################################



def MaximumLikelihoodEstmator(circuit_length, ones, zeros):

    grid = 50000
    epsilon = 1/grid
    domain = [0.0 + epsilon, np.pi/2 - epsilon] # to avoid zero

    def logL(theta):
        fval = 0
        for i in range(circuit_length):
            if i==0:
                fval += 2 * ones[i] * np.log( np.absolute(np.sin(theta)) )
                fval += 2 * zeros[i] * np.log( np.absolute(np.cos(theta)) )
            else:
                fval += 2 * ones[i] * np.log( np.absolute(np.sin((2 * (2**(i-1)) + 1) * theta)) )
                fval += 2 * zeros[i] * np.log( np.absolute(np.cos((2 * (2**(i-1)) + 1) * theta)) )
        return -fval # to compute maximum

    return brute(logL, [domain], Ns=grid)[0]





def mlae(n: int, m: int, b:float, shots: int, backend, printData = True) -> [list, list, list]:

    results = []
    one_hits = []
    zero_hits = []
    circuits = makeMLQAECircuits(n, m, b)
    #print('circuits length: ', len(circuits))
    #for i in range(m+1):
    #    print('circuits[',i,'] depth: ', circuits[i].depth())

    jobs = [execute(circuits[k], backend, shots=shots) for k in range(len(circuits))]
    results = [jobs[k].result() for k in range(len(jobs))]
    counts = [results[k].get_counts(circuits[k]) for k in range(len(circuits))]

    for c in counts:
        ones = c.get('1', 0)
        one_hits += [ones]
        zero_hits += [shots-ones]
        
        
    #for j in range(len(counts)):
    #    ones = counts[j].get('1', 0)
    #    one_hits += [ones]
    #    zero_hits += [shots-ones]

    
    num_oracle = (2*(2**m - 1) + (m+1)) * shots
    
    if printData:
        print('Oracle Queries: ', num_oracle)
    
    return one_hits, zero_hits, circuits



def stat_mlae(n: int, m: int, b:float, shots: int, trial: int, backend, true_a: float) -> [list, list, list, list]:
    
    theta = np.arcsin(np.sqrt(true_a))

    est_theta = []
    est_a = []
    rel_err_theta = []
    rel_err_a = []
    

    for i in range(trial):
        ones, zeros, _ = mlae(n, m, b, shots, backend, printData=False)
        #print('ones length', len(ones))
        #print('zeros length', len(zeros))
        th = MaximumLikelihoodEstmator(m+1, ones, zeros)
        a = np.sin(th)**2
        est_theta += [th]
        est_a += [a]
        rel_err_theta += [relativeError(theta, th)]
        rel_err_a += [relativeError(true_a, a)]

        
    return est_theta, est_a, rel_err_theta, rel_err_a


