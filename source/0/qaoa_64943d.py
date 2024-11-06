# https://github.com/soosub/bachelor-thesis/blob/7da3447e1fd77a9d94f79b7939dfc952d4f0e11e/Implementation/Zhou_1/QAOA.py
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 17:18:10 2020

@author: joost
"""
import qiskit 
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import scipy

plt.close('all')

def objective(G, x):
    '''Calculating the value of a cut x on graph G, 
    where x is represented by a numpy array of 1s and 0s
    '''
    w = nx.to_numpy_array(G)
    spin = (-1)**x # bijection from 0,1 to 1,-1
    
    return 0.5*sum([w[i,j]*(1-spin[i]*spin[j]) for i,j in G.edges]) 

def create_circuit(G, gamma, beta):
    '''Creating QAOA circuit for given angles'''
    # Setting p (and checking if beta and gamma are of the same size)
    if len(beta) == len(gamma):
        p = len(beta)
    else:
        raise ValueError("The parameter array beta must have the same length as parameter gamma")
    
    n = len(G)
    circ = qiskit.QuantumCircuit(n)
    
    for i in range(n):
        circ.h(i)
        
    for i in range(p):
        for a,b in G.edges:
            circ.cx(a,b)
            circ.rz(-gamma[i],b) #PLUS or MINUS! Does it matter? We optimize anyway - it does affect the range of gamma and the patterns
            circ.cx(a,b)
        circ.barrier()
        for a in G.nodes:
            circ.rx(beta[i], a)
    
    circ.measure_all()
    
    return circ

def expectation(G, gamma, beta, backend, n_samples = 1024):
    '''Evaluates expectation for given graph and angles'''
    
    # create circuit and run it n_shots times
    counts = sample(G, gamma, beta, backend, n_samples = n_samples)
    
    def bitstring_to_array(x):
        return np.array([int(d) for d in str(x)])
    
    avg = sum([counts[x] * objective(G,bitstring_to_array(x)) for x in counts])/n_samples
    
    return avg

def find_local_angles(G, gamma, beta, backend, n_samples = 1024, optimizer = 'Nelder-Mead', optimizer_options = None):
    '''Find local optimum beginning from gamma, beta (these are overwritten)
    Note that (gamma,beta) is the initial point and are overwritten when this method is run
    '''    

    def fun(angles):
        ''' The objective function to be optimised by varying the angles
        Note, we want to maximize the expectation <Hc>, this is equivalent to 
        minimizing - <Hc>'''
        new_gamma = angles[:len(angles)//2]
        new_beta = angles[len(angles)//2:]
        
        # maximize f = minimize -f 
        return -expectation(G, new_gamma, new_beta, backend, n_samples) 
    
    x0 = gamma+beta
    result = scipy.optimize.minimize(fun, x0, method= optimizer)
    
    # finding angles
    angles = result.x
    new_gamma = angles[:len(angles)//2]
    new_beta = angles[len(angles)//2:]
    return new_gamma, new_beta
  
def find_local_amplitudes(G, u, v, p, backend, n_samples = 1024, optimizer = 'Nelder-Mead', optimizer_options = None):
    '''Find local optimum beginning from u, v (these are overwritten)
    Note that (u,v) is the initial point and are overwritten when this method is run
    '''    

    def fun(amplitudes):
        ''' The objective function to be optimised by varying the angles
        Note, we want to maximize the expectation <Hc>, this is equivalent to 
        minimizing - <Hc>'''
        new_u, new_v  = amplitudes[:len(amplitudes)//2], amplitudes[len(amplitudes)//2:]
        new_gamma, new_beta = translate_fourier_to_angles(new_u, new_v, p)
        
        # maximize f = minimize -f 
        return -expectation(G, new_gamma, new_beta, backend, n_samples) 
    
    x0 = u+v
    result = scipy.optimize.minimize(fun, x0, method= optimizer)
    
    # finding angles
    amplitudes = result.x
    new_u = amplitudes[:len(amplitudes)//2]
    new_v = amplitudes[len(amplitudes)//2:]
    return new_u, new_v
    
def sample(G, gamma, beta, backend, n_samples = 1024, plot_histogram = False):
    circ = create_circuit(G,beta,gamma)
    result = qiskit.execute(circ, backend = backend, shots = n_samples).result()
    counts = result.get_counts()
    
    if plot_histogram:
        from qiskit.tools.visualization import plot_histogram
        plot_histogram(counts)
        
    return counts

def get_angles_INTERP(G, p, backend, initial_gamma = [1.2], initial_beta = [2]):
    '''
    Finds quasi-optimal angles for a given graph G
    Iteratively increase p, using an interpolation-based strategy
    to find initial points for optimization
    '''
    
    def INTERP_next_angles(angles):
        '''Takes in a set of p angles and returns a new set of p+1 angles using the INTERP method'''
        
        assert len(gamma) == len(beta), "Both lists of parameters gamma, beta must be of the same length"
        
        p = len(angles)
        
        new_angles = [angles[0]] # i = 1, the first angles coincide
        for i in range(2, p+1):
            new_angles.append( (i-1)/p*angles[i-2] + (p-i+1)/p*angles[i-1]) # i in {2, ... , p}, interpolation 
        new_angles.append(angles[-1]) # i = p+1, the last angles coincide
            
        return new_angles

    # p = 1 (check this - in Zhou et al. gamma in [-pi/2,pi/2) for udR graphs and beta in [-pi/4, pi/4]]) in general
    
    # find first set of angles
    gamma,beta = find_local_angles(G,initial_gamma,initial_beta,backend)
    
    for i in range(p-1):
        gamma_ansatz,beta_ansatz = INTERP_next_angles(gamma), INTERP_next_angles(beta) # new initial points
        print("ansatz gamma, beta", gamma_ansatz, beta_ansatz)
        gamma,beta = find_local_angles(G,gamma_ansatz,beta_ansatz,backend) # new (local) optima  
        print("local optima", gamma, beta)
    
    return gamma, beta

def get_angles_FOURIER(G, p, q, backend, R=0, initial_u = [0], initial_v = [0]): # other parameters? n_shots
    '''
    Finds quasi-optimal angles for a given graph G using the FOURIER method
    Iteratively increase p, using the FOURIER method, to find initial points for optimization of the next step
    The amplitudes of the respective frequencies are translated to angles
    '''
    raise NotImplementedError("Not properly implented, especially with perturbations")
        
    def FOURIER_next_amplitudes(amplitudes):
        '''Takes in a set of amplitudes and returns a new set of amplitudes using the FOURIER(q,R) method'''
        if q == p:
            return amplitudes+[0]
        else:
            return amplitudes
    
    u, v = find_local_amplitudes(G, initial_u,initial_v,p,backend) # other parameters as well?
    for i in range(1,p):
        u_ansatz,v_ansatz = FOURIER_next_amplitudes(u,v,i) # new initial points
        print("ansatz u, v", u_ansatz, v_ansatz)
        u,v = find_local_amplitudes(G,u_ansatz,v_ansatz,p,backend) # new (local) optima  
        print("local optima", u, v)
    
    return translate_fourier_to_angles(u, v, p)
    
    
def translate_fourier_to_angles(u, v, p):
    '''Calculates gamma, beta from their fourier representation u, v'''
    assert len(u) == len(v), 'length u and v should be equal'
    assert p >= 1, 'p must be a positive integer'
    
    q = len(u)
    k = np.arange(1,q+1)
    
    gamma = beta = [None]*p
    
    for i in range(1,p+1):
        # note that the index i starts from 0 as opposed to 1
        gamma[i-1] = np.sum(u*np.sin((k-1/2)*(i-1/2)*np.pi/p))
        beta[i-1] =  np.sum(v*np.cos((k-1/2)*(i-1/2)*np.pi/p))
        
    return gamma, beta

