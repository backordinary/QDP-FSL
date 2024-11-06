# https://github.com/soosub/bachelor-thesis/blob/7da3447e1fd77a9d94f79b7939dfc952d4f0e11e/Implementation/Qiskit/data_acquisition_noiseless.py
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 13:48:49 2020

@author: joost

Per experiment it returns a dictionary with the following keys
    - energy
    - time (in seconds)
    - iterations
    - max-cut objective
    - solution
    - solution objective
  
Multiple experiments are run and the data will be collected in a dataframe.
"""

# imports
from qiskit import Aer
import qiskit.aqua.components.optimizers as optimizers
from experiment import run_experiment
import pandas as pd
import my_graphs

# Defining graphs and (classical) optimizer
graphs = {'diamond': my_graphs.diamond(), 
          'butterfly': my_graphs.butterfly(), 
          'regular(3,4)': my_graphs.random_regular(3, 4)}
optimizer = optimizers.NELDER_MEAD()

# Backend
backend = Aer.get_backend('qasm_simulator')               #simulator

# Producing the data
repetitions = 10
p_min, p_max = 1, 1

for key, value in graphs.items():
    graph_name = key
    G = value
    
    for p in range(p_min,p_max + 1):
        output = pd.DataFrame()
        
        # specifications
        specs = {'graph': graph_name, 'p': p, 'optimizer': 'NELDER_MEAD', 'backend': backend.name(), 'noise': False, 'topology':False}
    
        for i in range(repetitions):
            # Running an experiment
            print("\nExperiment "+str(i+1)+"/"+str(repetitions)+", p = "+str(p))
            
            result = run_experiment(G, p, optimizer, backend, 
                                    print_result = True
                                    )
            result.update(specs) #adding specifications to the dictionary
            
            output = output.append(result, ignore_index=True)
        
        title = 'qiskit-'+ specs['graph'] +'-'+specs['backend']+'-noiseless-p'+str(p)
        output.to_csv('data/'+title+'.csv')








