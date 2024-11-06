# https://github.com/FredericSauv/qc_optim/blob/d30cd5d55d89a9ce2c975a8f8891395e94e763f0/_old/studies/h3_chain_sharedBO.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 16:02:02 2020

@author: kiran
"""

import os
import sys
import time
import copy
import joblib
import numpy as np
import qiskit as qk
import qcoptim as qc
import joblib as jbl
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from qiskit.ignis.mitigation.measurement import CompleteMeasFitter
# from qextras.chip import find_best_embedding_circuit_backend,embedding_to_initial_layout

pi = np.pi
# ------------------------------------------------------
# General Qiskit related helper functions
# ------------------------------------------------------

try:
    max_iter = sys.argv[1]
    max_iter = int(max_iter)
except:
    max_iter = 2
shape = (2, 2)
positions = np.linspace(0.2, 2, shape[0])





fname = 'H3_non_periodic_fewerprms.dmp'
if fname not in os.listdir():
    raise FileNotFoundError("This test assumes you have circ dmp file: 'h3_circs_prmBool.dmp'")
data = jbl.load(fname)
ansatz = qc.ansatz.AnsatzFromQasm(data[0]['qasm'], data[0]['should_prm'])
inst = qc.utilities.FakeQuantumInstance()

wpo_list = [qc.utilities.get_H_chain_qubit_op([dx1,dx2]) for dx1 in positions for dx2 in positions]
wpo_list = qc.utilities.enforce_qubit_op_consistency(wpo_list)
cost_list = [qc.cost.CostWPOquimb(ansatz, inst, ww) for ww in wpo_list]

#%%

def run_bo(nb_init, nb_iter):
    
    domain = np.array([(0, 2*pi) for i in range(cost_list[0].ansatz.nb_params)])
    bo_args = qc.utilities.gen_default_argsbo(f=lambda: 0.5,
                                              domain=domain,
                                              nb_init=nb_init,
                                              eval_init=False)
    bo_args['acquisition_weight'] = 3
    bo_args_vec = []
    for scaling in find_iter_vals():
        bo_args['nb_iter'] = int((nb_iter-1)*scaling  + 1)
        bo_args_vec.append(copy.deepcopy(bo_args))

        
    runner = qc.optimisers.ParallelRunner(cost_list,
                                          qc.optimisers.MethodBO,
                                          optimizer_args = bo_args_vec,
                                          share_init = True,
                                          method = '2d')

      
    runner.next_evaluation_circuits()
    print('there are {} init circuits'.format(len(runner.circs_to_exec)))
    
    t = time.time()
    results = inst.execute(runner.circs_to_exec,had_transpiled=True)
    print('took {:2g} s to run inits'.format(time.time() - t))
    
    t = time.time()
    runner.init_optimisers(results)
    print('took {:2g} s to init the {} optims from {} points'.format(time.time() - t, shape[0]**2, bo_args['initial_design_numdata']))
    
    for ii in range(nb_iter):
        t = time.time()
        runner.next_evaluation_circuits()
        print('took {:2g} s to optim acq function'.format(time.time()  - t))
    
        t = time.time()
        results = inst.execute(runner.circs_to_exec,had_transpiled=True)
        print('took {:2g} s to run circs'.format(time.time()  - t))
    
        t = time.time()
        runner.update(results)
        print('took {:2g} s to run {}th update'.format(time.time() - t, ii))


    energies = [min(runner.optim_list[ii].optimiser.Y.squeeze()) for ii in range(np.prod(shape))]
    energies = np.array(energies).reshape(shape)
    
    return energies

def find_iter_vals():
    mat = np.ones(shape)
    mask = np.ones((3,3))
    mask[0,0],mask[2,0],mask[0,2],mask[2,2]=0,0,0,0
    
    calls_per_iter = convolve2d(mat, mask, mode='same')
    return calls_per_iter.reshape(np.prod(shape))
#%%

save_dict = {'positions':positions}
nb_monte_carlo = 2
iter_vec = [max_iter + 4]
for itter in iter_vec:
    print(itter)
    this_data = [run_bo(1,itter) for ii in range(nb_monte_carlo)]
    save_dict.update({str(itter):this_data})

fname = 'iter_'+str(iter_vec[0])+ '_carlo_' +str(nb_monte_carlo) +'.dmp'
jbl.dump(save_dict, fname)
    
    
    
# runner.next_evaluation_circuits()
# results = inst.execute(runner.circs_to_exec,had_transpiled=True)
# runner.init_optimisers(results)


# for ii in range(bo_args['nb_iter']):

#     runner.next_evaluation_circuits()

#     results = inst.execute(runner.circs_to_exec,had_transpiled=True)
#     runner.update(results)

