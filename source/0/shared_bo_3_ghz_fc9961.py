# https://github.com/FredericSauv/qc_optim/blob/d30cd5d55d89a9ce2c975a8f8891395e94e763f0/_old/studies/shared_BO_3_ghz.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 8 09:15:32 2020
@author: Kiran
"""

# 
#%% IMPORTS

import copy
import dill
import time
import numpy as np
import pandas as pd
import qiskit as qk
import seaborn as sns
import matplotlib.pyplot as plt
from qcoptim import ansatz as az
from qcoptim import cost as cost
from qcoptim import utilities as ut
from qcoptim import optimisers as op



#%% DEFAULTS 

# ======================== /
# Defaults and global objects
# ======================== /
pi= np.pi
NB_SHOTS_DEFAULT = 1024
OPTIMIZATION_LEVEL_DEFAULT = 0
NB_TRIALS = 2
NB_CALLS = 300
NB_IN_IT_RATIO = 0.5001024
NB_OPT_VEC = [1,2]
SAVE_DATA = False

nb_init_vec = []
nb_iter_vec = []
for opt in NB_OPT_VEC:
    nb_init_vec.append(round((NB_CALLS * NB_IN_IT_RATIO) / opt))
    nb_iter_vec.append(round((NB_CALLS * (1 - NB_IN_IT_RATIO)) / opt))
    print(opt * (nb_init_vec[-1] + nb_iter_vec[-1]))
# for ii in range(1,len(NB_OPT_VEC)):
#     nb_init_vec[ii] = nb_init_vec[0] - 2*ii

simulator = qk.Aer.get_backend('qasm_simulator')
inst = qk.aqua.QuantumInstance(simulator,
                               shots=NB_SHOTS_DEFAULT,
                               optimization_level=OPTIMIZATION_LEVEL_DEFAULT)
Batch = ut.Batch(inst)
np.random.seed(int(time.time()))



#%% COST PARAMETERS AND OPTIMISER
# ======================== /
# Generate ansatz and cost here
# ======================== /
x_sol = np.pi/2 * np.array([1.,1.,2.,1.,1.,1.])
anz_hard = az.AnsatzFromFunction(az._GraphCycl_6qubits_24params)
anz_medi = az.AnsatzFromFunction(az._GraphCycl_6qubits_12params)
anz_easy = az.AnsatzFromFunction(az._GraphCycl_6qubits_6params)
anz_rand = az.RandomAnsatz(6, 2)

anz_ghz = az.AnsatzFromFunction(az._GHZ_3qubits_6_params_cx0)

anz = anz_ghz
cst = cost.GHZPauliCost(anz, inst, invert = True)


# ======================== /
#  Default BO optim args
# ======================== /
bo_args = ut.gen_default_argsbo(f=lambda x: .5, 
                                domain= [(0, 2*np.pi) for i in range(anz.nb_params)], 
                                nb_init=0,
                                eval_init=False)


# ======================== /
# Create runners
# ======================== /
# df = pd.DataFrame()
runner_dict = {}
for trial in range(NB_TRIALS):
    for opt, init, itt in zip(NB_OPT_VEC, nb_init_vec, nb_iter_vec):
        bo_args['nb_iter'] = itt*opt + init
        bo_args['initial_design_numdata'] = init
        runner = op.ParallelRunner([cst]*opt, 
                                   op.MethodBO, 
                                   optimizer_args = bo_args,
                                   share_init = True,
                                   method = 'shared')        
        runner_dict[(opt,trial)] = [runner, itt]


#%% RUN OPTIMISER
# ======================== /
# Init runners
# ======================== /
print('Init runners')
t = time.time()
Batch = ut.Batch(inst)
for run, _ in runner_dict.values():
    run.next_evaluation_circuits()
    Batch.submit(run)
temp = len(Batch.circ_list)
Batch.execute()

for run, _ in runner_dict.values():
    Batch.result(run)
    run.init_optimisers()
    
print('took {} s to init from {} circuits'.format(round(time.time() - t), temp))

# ======================== /
# Run optimisation
# ======================== /
print("Running optims")
for ii in range(max(nb_iter_vec)):
    t = time.time()
    for opt, trial in runner_dict.keys():
        run, max_itt = runner_dict[(opt, trial)]
        if ii < max_itt:
            run.next_evaluation_circuits()
            Batch.submit(run)
            
    temp = len(Batch.circ_list)
    Batch.execute()
    
    for opt, trial in runner_dict.keys():
        run, max_itt = runner_dict[(opt, trial)]
        if ii < max_itt:
            Batch.result(run)
            run.update()
    print('iter: {} of {} took {} s for {} circuits'.format(ii+1, max(nb_iter_vec), round(time.time() - t), temp))
    np.random.seed(int(time.time()))



# ======================== /
# Get results
# ======================== /
print('Generating results')
df = pd.DataFrame()
example_optim = {}
for ct, (opt, trial) in enumerate(runner_dict.keys()):
    run, _ = runner_dict[(opt, trial)]
    x_opt_pred = [opt.best_x for opt in run.optim_list]
    run.shot_noise(x_opt_pred, nb_trials=5)
    Batch.submit_exec_res(run)
    bopt_lines = run._results_from_last_x()
    
    m = np.mean(bopt_lines, axis = 1)
    v = np.std(bopt_lines, axis = 1)
    dat = [min(m), v[m == min(m)][0], opt, trial, nb_init_vec[NB_OPT_VEC.index(opt)], nb_iter_vec[NB_OPT_VEC.index(opt)]]
    df_temp = pd.DataFrame([dat], columns = ['mean', 'std', 'nb_opt', 'trial', 'nb_init', 'nb_iter'], index=[ct])
    df = df.append(df_temp)
    example_optim[(opt, trial)] = run.optim_list[0].optimiser 


# ======================== /
# Save data
# ======================== /
fname = str(cst.__class__).split('cost.')[1].split("'")[0]
fname = fname + '_{}calls_{}ratio'.format(NB_CALLS,NB_IN_IT_RATIO).replace('.', 'p') + '.pkl'
if SAVE_DATA:
    dict_to_dill = {'df':df,
                    'anz':anz,
                    'example_optim':example_optim,
                    'NB_IN_IT_RATIO':NB_IN_IT_RATIO,
                    'NB_CALLS':NB_CALLS,
                    'NB_SHOTS_DEFAULT':NB_SHOTS_DEFAULT}
    with open(fname, 'wb') as f:                                                                                                                                                                                                          
        dill.dump(dict_to_dill, f) 



#%% LOAD / PLOT DATA
# ========================= / 
# Files: 
#        1 optims         
#        fname = 'GHZPauliCost_200calls_0p5001024ratio.pkl'
#        fname = 'GHZPauliCost_200calls_0p50064ratio.pkl'
#        fname = 'GHZPauliCost_200calls_0p5002048ratio.pkl'
#
#        4 optims 
#        fname = 'GHZPauliCost_150calls_0p50064ratio.pkl'        # 4 optim 15 trials 64 shots/call
#        fname = 'GHZPauliCost_150calls_0p5001024ratio.pkl'    # 4 optims 15 trial 1024 shots/call
#        fname = 'GHZPauliCost_150calls_0p5002048ratio.pkl'
#
#        fname = 'GHZWitness2Cost_150calls_0p5001024ratio.pkl' 
#        fname = 'GHZWitness2Cost_150calls_0p5002048ratio.pkl' 
#        fname = 'GHZWitness2Cost_150calls_0p50064ratio.pkl'
# 
#         1 optims 
#         fname = 'GHZPauliCost_180calls_0p500128ratio.pkl'
#         fname = 'GHZPauliCost_180calls_0p5001024ratio.pkl'
#         fname = 'GHZPauliCost_180calls_0p5002048ratio.pkl'
#
#         1 optim, proper update weights
#         fname = 'GHZPauliCost_100calls_0p5002048ratio.pkl'
# ========================= /
if SAVE_DATA: 
    import copy
    import dill
    import time
    import numpy as np
    import pandas as pd
    import qiskit as qk
    import seaborn as sns
    import matplotlib.pyplot as plt
    from qcoptim import ansatz as az
    from qcoptim import cost as cost
    from qcoptim import utilities as ut
    from qcoptim import optimisers as op
    with open(fname, 'rb') as f:
        data = dill.load(f)
        df = data['df']
        anz = data['anz']
        example_optim = data['example_optim']
        NB_CALLS = data['NB_CALLS']
        NB_SHOTS_DEFAULT = data['NB_SHOTS_DEFAULT']
        NB_IN_IT_RATIO = data['NB_IN_IT_RATIO']
        NB_TRIALS = df.trial.max() + 1
        NB_OPT_VEC = sorted(df.nb_opt.unique())
        nb_init_vec = sorted(df.nb_init.unique(),reverse=True)
        nb_iter_vec = sorted(df.nb_iter.unique(),reverse=True)



# ======================== /
# Plot results
# ======================== /
sns.set()
plt.close('all')

f = plt.figure(1, figsize=(10,5))
axes = f.subplots(1, 3, sharey=True)
for ii in range(len(df)):
    m = df.iloc[ii]['mean']
    v = df.iloc[ii]['std']
    t = df.iloc[ii]['trial']
    o = df.iloc[ii]['nb_opt']
    axes[0].errorbar(o + 0.1*t/NB_TRIALS, m, yerr = v, fmt = 'r.', label='t {}'.format(t))
axes[0].set_title('Shot noise ({} shots/circ)'.format(NB_SHOTS_DEFAULT))
axes[0].set_ylabel('Cost ' + fname)
axes[0].set_xlabel('nb optimisers')


# plt.figure(4)
# for ii in range(len(df)):
#     m = df.iloc[ii]['mean'] 
#     v = df.iloc[ii]['std']
#     t = df.iloc[ii]['trial']
#     o = df.iloc[ii]['nb_opt']
#     plt.plot(o + 0.1*t/NB_TRIALS, np.log10(m), 'b.')
# plt.ylabel('log10 Cost')



sns.pointplot(data = df, x = 'nb_opt', y = 'mean', join=False, ax=axes[1])
axes[1].set_title('Optimiser noise')
axes[1].set_xlabel('nb optimisers')
axes[1].set_ylabel('cost')


sns.boxplot(data = df, x = 'nb_opt', y = 'mean', ax = axes[2])
axes[2].set_title('Optimiser noise')
axes[2].set_xlabel('nb optimisers')
axes[2].set_ylabel('cost')
f.show()




f = plt.figure(2, figsize=(5, 11))
axes = f.subplots(len(NB_OPT_VEC), 1, sharex=True)
axes = np.atleast_1d(axes)
for ii, opt in enumerate(NB_OPT_VEC):
    axes[ii].set_title('nb optims: {}'.format(opt))
    #axes[ii].set_ylim(df['mean'].min() - 0.1, df['mean'].max() + 0.1)

    if ii == int(len(NB_OPT_VEC) / 2):
        axes[ii].set_ylabel('Cost')
    for trial in range(NB_TRIALS):
        data = np.ravel(example_optim[(opt, trial)].Y)
        iter_data = data[nb_init_vec[ii]:]
        # iter_data = data
        sns.scatterplot(np.arange(1,len(iter_data)+1),  iter_data, ax=axes[ii])
axes[ii].set_xlabel('iter')
f.show()




f = plt.figure(3, figsize=(10, 5))
axes = f.subplots(1, len(NB_OPT_VEC), sharey=True, squeeze=False)
axes = np.ravel(axes)
for opt, trial in example_optim.keys():
    bo = example_optim[(opt,trial)]
    x = ut._diff_between_x(bo.X) 
    sns.lineplot(np.arange(1, len(x) + 1),  x, ax=axes[NB_OPT_VEC.index(opt)]) 
    axes[NB_OPT_VEC.index(opt)].set_title('nb opt: {}'.format(opt))
    axes[NB_OPT_VEC.index(opt)].set_xlabel('iter')
f.show()
axes[0].set_ylabel('x_{i + 1} - x_{i}')
[ax.set_xlabel('iter') for ax in axes]
f.show()




# plt.figure(4)
# for ii in range(len(df)):
#     m = df.iloc[ii]['mean'] 
#     v = df.iloc[ii]['std']
#     t = df.iloc[ii]['trial']
#     o = df.iloc[ii]['nb_opt']
#     plt.plot(o + 0.1*t/NB_TRIALS, np.log10(m), 'b.')
# plt.ylabel('log10 Cost')

print('NB_SHOTS_DEFAULT: ' + str(NB_SHOTS_DEFAULT))
print('NB_TRIALS: ' + str(NB_TRIALS))
print('NB_OPT_VEC: ' + str(NB_OPT_VEC))