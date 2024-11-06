# https://github.com/soosub/bachelor-thesis/blob/7da3447e1fd77a9d94f79b7939dfc952d4f0e11e/Implementation/Qiskit/data_acquisition.py
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
from qiskit import IBMQ, Aer
import qiskit.aqua.components.optimizers as optimizers
from experiment import run_experiment
import pandas as pd
import my_graphs
from qiskit.providers.aer.noise import NoiseModel

# Defining graphs and (classical) optimizer
graphs = {'regular(3,4)': my_graphs.random_regular(3, 4)}
optimizer = optimizers.NELDER_MEAD()

# Backend
provider = IBMQ.load_account()
backend = provider.get_backend('ibmq_vigo')           #simulated
 
# Noise model et cetera
noise_model = NoiseModel.from_backend(backend)                  # Get noise model from backend
coupling_map = backend.configuration().coupling_map             # Get coupling map from backend
basis_gates = noise_model.basis_gates                           # Get basis gates from noise model

# Producing the data
repetitions = 1
p_min, p_max = 1, 1

for key, value in graphs.items():
    graph_name = key
    G = value
    
    for p in range(p_min,p_max + 1):
        output = pd.DataFrame()
        
        # specifications
        specs = {'graph': graph_name, 'p': p, 'optimizer': 'NELDER_MEAD', 'backend':'qasm_simulator', 'simulated backend': backend.name(), 'noise': True, 'topology':True}
    
        for i in range(repetitions):
            # Running an experiment
            print("\nExperiment "+str(i+1)+"/"+str(repetitions)+", p = "+str(p))
            
            result = run_experiment(G, p, optimizer, Aer.get_backend('qasm_simulator'), 
                                    print_result = True,
                                    noise_model = noise_model,
                                    coupling_map = coupling_map,
                                    basis_gates = basis_gates
                                    )
            result.update(specs) #adding specifications to the dictionary
            
            output = output.append(result, ignore_index=True)
        
        file_title = 'qiskit-'+ specs['graph'] +'-'+specs['simulated backend']+'-on-'+specs['backend']+'-p'+str(p)
        output.to_csv('data/'+file_title+'.csv')








