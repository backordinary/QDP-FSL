# https://github.com/EeshGupta/VQE_Research/blob/98a0674eca203119b0c9ae358ce0379ba0a869fa/Coding%20Runs/Experiments/November/Pulse%20and%20Gate%20Comparison%20VQE%20H2%20(Nov%2017)/ResultVQE_funcs.py
import numpy as np
import random
from qiskit.tools.monitor import job_monitor
from qiskit import QuantumCircuit, execute, Aer, IBMQ, IBMQ
from qiskit import *

def binaryToDecimal(binary):  
    binary1 = binary 
    decimal, i, n = 0, 0, 0
    while(binary != 0): 
        dec = binary % 10
        decimal = decimal + dec * pow(2, i) 
        binary = binary//10
        i += 1
    return decimal

def dictToList(dicty): 
    '''
    Converts dictionary to a list of keys appearing [frequency] number of times
    '''
    keys = dicty.keys()
    listy = []
    
    for key in keys: 
        listy += [key for i in range(dicty[key])]
    
    return listy

def sampleExpecVal(samp_zsis, samp_xx, Hamiltonian, Hamiltonian_weights):
    '''
    Input: count corresponding to zz, zi, etc..., count corresponding to xx
    Output: expectation value of the sample
    '''
    
    #Hamiltonian = """II IZ ZI  ZZ XX"""
    Hamiltonian = ['II', 'IZ', 'ZI', 'ZZ', 'XX' ]
    #Hamiltonian_weights = [-1.053, 0.395, -0.395, -0.011, 0.181]
    Hamiltonian_eig = []
    
    for hammy in Hamiltonian: 
        if (hammy!= 'XX'): 
            Hamiltonian_eig+= [countToEig(samp_zsis, hammy)]
        else:
            Hamiltonian_eig+= [countToEig(samp_xx, hammy)]
            
    #combining eigvals of local hamiltonians 
    energy = np.dot(Hamiltonian_weights, Hamiltonian_eig)
    
    #add in the shift (nuclear repulsion energy)
    shift = 0.7151043390810812
    
    return energy + shift

import numpy as np
from qiskit.quantum_info import Pauli 

def countToEig(count, matrix): 
    '''
    Input: count (string), matrix (2 bit string)
    Output: eigval corresponding to that count
    '''
    #print(count)
    #print(matrix)
    #general matrices 
    x = [[0, 1],[1, 0]]
    z = [[1, 0], [0, -1]]
    i = [[1, 0],[0,1]]
    #parsing matrices
    matrices = []
    for mat in matrix: 
        if (mat== 'X'):
            matrices.append(x)
        elif(mat == 'Z'):
            matrices.append(z)
        elif(mat == 'I'): 
            matrices.append(i)
        else: 
            print('Error parsing matrices')
    first = matrices[0]
    second = matrices[1]
#     print('Matrices are ')
#     print(first)
#     print(second)
    
    #computing eigenvalue of kron(first, second)
    v, w = np.linalg.eig(np.kron(first, second))
    #convert count to dec
    count = binaryToDecimal(int(count))
    return v[count]

def expecValForSamples(counts_zsis, counts_xx, Hamiltonian, hammy_weights):
    '''
    Input: counts [dict] for zz,iz, etc. , counts[dict] for xx
    Output: List of expecVal for all the samples
    '''
    
    #convert dict to list
    list_zsis = dictToList(counts_zsis)
    list_xx = dictToList(counts_xx)
    
    #extract samples from list and compute expec val
    expec_vals = []
    while(len(list_zsis)!= 0 and len(list_xx)!= 0): 
        index1 = random.randrange(0, len(list_zsis))
        index2 = random.randrange(0, len(list_xx))
        
        exp_val= sampleExpecVal(list_zsis[index1], list_xx[index2], Hamiltonian, hammy_weights)
        expec_vals.append(exp_val)
        
        #removing those items from list 
        list_zsis.pop(index1)
        list_xx.pop(index2)
    return expec_vals

def addDicts(listy): 
    '''
    Input: list of dicts of counts 
    Output: combining all dicts into one dict, returning that
    '''
    keys = ['00', '11', '10', '01']
    master = {}
    
    for key in keys: 
        #initializing master at that key
        master[key] = 0
        
        #now adding up all dictys[key]
        for dicty in listy: 
            try:
                master[key] += dicty[key]
            except KeyError: 
                continue
    return master


def evaluator(circuits, samples, Hamiltonian, hammy_weights, machine):
    """
    Input: circuits, noise model to run on , weights of the local hammys
    Output: expectation value (energy)
    
    """
    
    job = execute(circuits, backend = machine, shots = samples)
    job_monitor(job)
    counties = job.result().get_counts()
    
    #print('Adding dicts done')
    #computing expectation values
    expec_vals = expecValForSamples(counties[0], counties[1], Hamiltonian, hammy_weights)
    #print('Computing expec vals done')
    
    mean = np.mean(expec_vals)
    #std = np.std(expec_vals)
        
    return mean
    