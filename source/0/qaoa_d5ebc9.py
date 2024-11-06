# https://github.com/soosub/bachelor-thesis/blob/7da3447e1fd77a9d94f79b7939dfc952d4f0e11e/Implementation/Zhou_3/QAOA.py
# -*- coding: utf-8 -*-
"""
Created on Tue June 23

@author: joost
"""

import qiskit 
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import scipy
from collections import defaultdict

class QAOA:   
    def __init__(self, G, n_samples = None, backend = qiskit.Aer.get_backend('qasm_simulator'),
                 optimizer = 'COBYLA', optimizer_options = None, constrained = False):
        '''Note that optimizer and optimizer_options are not used for PSO optimization'''
        self.G = G
        self.backend = backend
        
        # storing angle sets
        self.gammas_list = []
        self.betas_list = []
        
        # optimizer options
        self.optimizer = optimizer
        self.optimizer_options = optimizer_options
        self.constrained = False
        
        # number of samples for expectation estimation and final result sampling
        if not n_samples == None:
            self.n_samples = n_samples
        else:
            self.n_samples = len(G.edges)**2 # This does not work properly when running optimization
        
        # turning unweighted graphs into weighted graph with weight wij = 1
        for (u, v) in G.edges():
            if G.edges[u,v] == {}:
                G.edges[u,v]['weight'] = 1
    
        
    def objective(self, x):
        '''Calculating the value of a cut x on graph G, 
        where x is represented by a numpy array of 1s and 0s
        '''     
        return sum(w['weight'] for i, j, w in self.G.edges(data=True) if x[i] != x[j]) 
    
    def create_circuit(self, gamma, beta):
        '''Creating QAOA circuit for given angles'''
        # Setting p (and checking if beta and gamma are of the same size)
        if len(beta) == len(gamma):
            p = len(beta)
        else:
            raise ValueError("The parameter array beta must have the same length as parameter gamma")
        
        G = self.G # graph
        
        n = len(G)
        circ = qiskit.QuantumCircuit(n)
        
        for i in range(n): # superposition state using hadamards
            circ.h(i)
            
        # watch out for the weights! :( - https://stackoverflow.com/questions/62538230/how-to-fix-inconsistent-labelling-of-edges-nodes-in-pythons-networkx
        for i in range(p): 
            for a,b, w in G.edges(data=True): # U(C,gamma) gates for every edge
                circ.cx(a,b)
                circ.rz(-gamma[i]*w['weight'],b) # See paper by Crooks (or it work out yourself :) 
                circ.cx(a,b)
            for a in G.nodes: # U(B, beta) gates for every node
                circ.rx(2*beta[i], a)
        
        circ.measure_all()
        
        return circ
        
    def expectation(self, gamma, beta, n_samples = None):
        '''Evaluates expectation for given graph and angles'''
    
        # create circuit and run it self.n_samples times
        if n_samples == None:
            n_samples = n_samples
        else:
            n_samples = self.n_samples
        counts = self.sample(gamma, beta, n_samples)
        avg = sum([counts[x] * self.objective(x) for x in counts])/n_samples
        
        return avg
    
    def constraint(p, g_min, g_max, b_min, b_max):
        '''Creates an constraint object for scipy minimizer'''
        A = np.eye(2*p)
        lb = np.zeros(2*p)
        ub = np.zeros(2*p)
        
        g_max = 2*np.pi
        g_min = 0
        b_max = np.pi
        b_min = 0
        
        lb[:len(lb)//2] = g_min
        lb[len(ub)//2:] = b_min
        ub[:len(lb)//2] = g_max
        ub[len(ub)//2:] = b_max

        return scipy.optimize.LinearConstraint(A, lb, ub, keep_feasible=False)
    
    def find_local_angles(self, gamma, beta,
                          g_min = 0, g_max = np.pi, 
                          b_min = 0, b_max = np.pi/2):
        '''Find local optimum beginning from gamma, beta
        Note that (gamma,beta) is the initial point and is NOT overwritten when this method is run
        The bounds on the constraints, g_min, g_max, b_min, b_max only apply if self.constrained = True
        '''
        if len(gamma) == len(beta):
            p = len(gamma)
        else:
            raise Exception("length of gamma and beta must be ste same")
        
        def fun(angles):
            ''' The objective function to be optimised by varying the angles
            Note, we want to maximize the expectation <Hc>, this is equivalent to 
            minimizing - <Hc>'''
            new_gamma = angles[:len(angles)//2]
            new_beta = angles[len(angles)//2:]
            
            # maximize f = minimize -f 
            return -self.expectation(new_gamma, new_beta) 
        
        x0 = gamma+beta
        if self.constrained:
            result = scipy.optimize.minimize(fun, x0, method= self.optimizer, constraints = QAOA.constraint(p, g_min,g_max,b_min,b_max), options = self.optimizer_options)
        else:
            result = scipy.optimize.minimize(fun, x0, method= self.optimizer, options = self.optimizer_options)
        
        # finding angles
        angles = result.x
        new_gamma = angles[:len(angles)//2]
        new_beta = angles[len(angles)//2:]
        return new_gamma, new_beta
    
    def gridsearch(self, n_gamma = 100, n_beta = 100,
                           g_min = -np.pi, g_max = np.pi, b_min = -np.pi/4, b_max = np.pi/4,
                           plot_F = False, imshow_F = False, print_optimum = False):
        
        '''Does a gridsearch to find optimal angles at p = 1, additionally it returns the landscape F
        as gamma,beta, F'''
        from tqdm import tqdm
        
        g_grid = np.linspace(g_min, g_max, n_gamma)
        b_grid = np.linspace(b_min, b_max, n_beta)
        
        F = np.zeros([len(g_grid), len(b_grid)])
        
        from pyquil_expectation import expectation as expect
        def f(x,y):
            # This method is wrong, for the expectation landscapes I used the pyquil stuff
            return expect(self.G,[x],[y],self.n_samples)
            #return self.expectation([x], [y])
    
        for i in tqdm(range(len(g_grid))):
            for j in range(len(b_grid)):
                F[i][j] = f(g_grid[i], b_grid[j])
        
        result = np.where(F == np.amax(F))
        a      = list(zip(result[0],result[1]))[0]

        gamma_max  = g_grid[a[0]]
        beta_max   = b_grid[a[1]]
        
        if plot_F:
            # Plot the expetation value F1
            from   matplotlib import cm
            from   matplotlib.ticker import LinearLocator, FormatStrFormatter
            from math import ceil
            
            fig = plt.figure()
            ax  = fig.gca(projection='3d')
            
            g_grid, b_grid = np.meshgrid(g_grid, b_grid)
            
            F = F.transpose()
            F = np.flip(F,0)
            
            ax.plot_surface(g_grid, b_grid[::-1], F, cmap=cm.coolwarm, linewidth=0, antialiased=True)
            
            ax.set_zlim(0,ceil(np.amax(F)))
            ax.zaxis.set_major_locator(LinearLocator(3))
            ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
            
            # Labels
            ax.set_xlabel(r'$\gamma$')
            ax.set_ylabel(r'$\beta $')
            ax.set_zlabel('$F$')
                        
            plt.show()
        
        if imshow_F:            
            fig = plt.figure()
            
            plt.imshow(F, extent = [g_min , g_max, b_min , b_max])
            plt.colorbar()
            
            # Labels
            plt.xlabel(r'$\gamma$')
            plt.ylabel(r'$\beta $')
            
            # maximum
            plt.plot([gamma_max],[beta_max],'ro')
            
        if print_optimum:
            #The smallest parameters and the expectation can be extracted
            print('\n --- OPTIMAL PARAMETERS --- \n')
            print('The maximal expectation value is:  M1 = %.03f' % np.amax(F))
            print('This is attained for gamma = %.03f and beta = %.03f' % (gamma_max,beta_max))
            
        return gamma_max, beta_max, F
                
      
    def sample(self, gamma, beta, n_samples, plot_histogram = False):        
        circ = self.create_circuit(beta,gamma)
        result = qiskit.execute(circ, backend = self.backend, shots = n_samples).result()
        counts = result.get_counts()
        
        if plot_histogram:
            from qiskit.tools.visualization import plot_histogram
            plot_histogram(counts)
            
        return counts
    
    # Analysing samples    
    def counts_to_valueCounts(self, counts, plot_histogram = False):
        d2 = defaultdict(int)
        for k, v in counts.items():
            d2[self.objective(k)] += v
            
        if plot_histogram:
            from qiskit.tools.visualization import plot_histogram
            plot_histogram(d2)
            
        return d2
    
    def mean_valueCounts(valueCounts):
        return np.sum([v*k for k,v in valueCounts.items()])/np.sum([v for k,v in valueCounts.items()])
    
    def best_valueCounts(valueCounts):
        return np.max([k for k,v in valueCounts.items()])
    
    def most_sampled(self, counts):
        raise NotImplementedError("Most sampled not implemented yet")
            
    def solution(self, gamma, beta):
        counts = self.sample(gamma, beta, self.n_samples)
        v_counts = self.counts_to_valueCounts(counts)
        return QAOA.best_valueCounts(v_counts)
        
        
    # Fraction of possible partitions
    def print_fraction(self):
        print(self.n_samples / 2**(len(self.G)-1))        
    
    # Drawing
    def draw_cut(x):
        raise NotImplementedError
        
    def draw_graph(self):
        nx.draw(self.G)
        raise NotImplementedError("Should add labels (correctly!)")
        
    # Remove degeneracies
    def remove_degeneracies_list(gammas,betas, g_period= np.pi/2, b_period = 2*np.pi):
        '''Removes degeneracies from angle arrays
        Warning, the array changes during the process'''
        for i in range(len(gammas)):
            gammas[i], betas[i] = QAOA.remove_degeneracies(gammas[i],betas[i])
        return gammas, betas
        
    def remove_degeneracies(gamma,beta, g_period= np.pi/2, b_period = 2*np.pi):
        ''' Removes degeneracies from numeric angles gamma, beta
        Warning, removing degeneracies with the default periods only works for unweighted graphs (or integer weigted graphs)'''
        # Move all points into domain [0,2pi] x [0,pi/2]
        gamma = gamma % g_period
        beta = beta % b_period
        
        # Removing extra degeneracies using time reversal F(g,b) = F(-g,-b)
        if beta > b_period/2:
            gamma = g_period - gamma
            beta = b_period - beta
        return gamma,beta
    
        
   
    
