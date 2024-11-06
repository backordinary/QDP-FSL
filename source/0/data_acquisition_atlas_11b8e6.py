# https://github.com/soosub/bachelor-thesis/blob/7da3447e1fd77a9d94f79b7939dfc952d4f0e11e/Implementation/Qiskit/data_acquisition_atlas.py
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 17:08:16 2020

@author: joost
"""

# imports
from qiskit import Aer
import qiskit.aqua.components.optimizers as optimizers
from experiment import run_experiment
import pandas as pd
import networkx as nx
from classical import brute_force

# Defining classical optimizer necessary for QAOA
optimizer = optimizers.SPSA()

# Backend
backend = Aer.get_backend('qasm_simulator')               #simulator

# Producing the data
p_min, p_max = 1, 3

# Putting the whole atlas in one data frame
output = pd.DataFrame()

for i in range(3,len(nx.graph_atlas_g())):
    G = nx.graph_atlas(i)
    n = len(G.nodes)
    m = len(G.edges)
    d = m / (n*(n-1)/2) # density of graph - number of edges / number of possible edges
    
    optimum = brute_force(G)
    print("\nGraph "+str(i)+": n = "+str(n), ", m = "+str(m), ", d = "+str(d), "the optimum cut is "+str(optimum)+"\n")
    
    if m > 0: # Method does not work if there are no edges
        for p in range(p_min,p_max + 1):
            # specifications
            specs = {'graph_atlas index': i, 'p': p, 'optimizer': 'SPSA', 'backend': backend.name(), 'noise': False, 'topology':False, 'optimum': optimum}
        
            # Running an experiment
            print("\nGraph "+str(i), "p = "+str(p))
            result = run_experiment(G, p, optimizer, backend, 
                                    print_result = True,
                                    n_shots=1000
                                    )
            result.update(specs) #adding specifications to the dictionary
            
            # Writing output to file
            output = output.append(result, ignore_index=True)
            output.to_csv('data/qiskit-graph_atlas-nshots_1000.csv')








