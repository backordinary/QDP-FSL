# https://github.com/FredericSauv/qc_optim/blob/d30cd5d55d89a9ce2c975a8f8891395e94e763f0/_old/studies/shared_BO_randomXY_configuration_chain.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 8 09:15:32 2020
@author: Kiran
"""

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


# 

#%% DEFAULTS 

# ======================== /
# Defaults and global objects
# ======================== /
pi = np.pi
NB_SHOTS_DEFAULT = 1024
OPTIMIZATION_LEVEL_DEFAULT = 0
NB_TRIALS = 4
NB_CALLS = 500
NB_IN_IT_RATIO = 0.5001024
NB_SPINS = 3
NB_DEPTH = 2
NB_OPT_VEC = [1]
SAVE_DATA = True
NB_ANZ_SEED = 10
NB_HAM_SEED = 3
NB_CONFIGS = 1


nb_init_vec = [round(NB_CALLS * NB_IN_IT_RATIO)]
nb_iter_vec = [round(NB_CALLS * (1 - NB_IN_IT_RATIO) / NB_CONFIGS)]
simulator = qk.Aer.get_backend('qasm_simulator')
inst = qk.aqua.QuantumInstance(simulator,
                               shots=NB_SHOTS_DEFAULT,
                               optimization_level=OPTIMIZATION_LEVEL_DEFAULT)
Batch = ut.Batch(inst)
np.random.seed(int(time.time()))



#%% COST PARAMETERS AND OPTIMISER
# ======================== /NB_CONFIGS
# Generate ansatz and cost here
# ======================== /
anz = az.RegularU2Ansatz(NB_SPINS,
                         NB_DEPTH,
                         seed = NB_ANZ_SEED,
                         cyclic = True)
cstvec = []
hvec = []
h_config = []
if NB_CONFIGS > 1:
    alphaVec = np.logspace(0, 1, NB_CONFIGS)
else:
    alphaVec = [10]
for aa in alphaVec:
    hamiltonian = ut.gen_random_xy_hamiltonian(NB_SPINS,
                                               U = 0,
                                               J = 0,
                                               delta = 1,
                                               alpha = aa,
                                               seed = NB_HAM_SEED) / NB_SPINS
    hvec.append(hamiltonian)
    cst = cost.RandomXYCost(anz, inst, hamiltonian)
    cstvec.append(cst)
    h_config.append(round(aa, 2))

# ======================== /
#  Default BO optim args
# ======================== /
bo_args = ut.gen_default_argsbo(f=lambda x: .5,
                                domain= [(0, 2*np.pi) for i in range(anz.nb_params)],
                                nb_init=0,
                                eval_init=False)
bo_args['acquisition_weight'] = 20

# ======================== /
# Create runners
# ======================== /
df = pd.DataFrame()
runner_dict = {}
for trial in range(NB_TRIALS):
    for opt, init, itt in zip(NB_OPT_VEC, nb_init_vec, nb_iter_vec):
        bo_args['nb_iter'] = itt*len(h_config) + init
        bo_args['initial_design_numdata'] = init
        runner = op.ParallelRunner(cstvec*opt,
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
for run, itt in runner_dict.values():
    bo_args = run.optimizer_args
    x_init = ut.gen_params_on_subspace(bo_args,
                                       nb_ignore=12,
                                       nb_ignore_ratio=0.0)
    run._gen_circuits_from_params([x_init], inplace=True)
    Batch.submit(run)
    print(itt)
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

    dat = [m, v,
           [opt]*len(cstvec),
           [trial]*len(cstvec),
           [nb_init_vec[NB_OPT_VEC.index(opt)]]*len(cstvec),
           [nb_iter_vec[NB_OPT_VEC.index(opt)]]*len(cstvec),
           h_config,
           [NB_ANZ_SEED]*len(cstvec)]
    dat = np.array(dat).transpose()
    columns = ['mean', 'std', 'nb_opt', 'trial',
               'nb_init', 'nb_iter', 'h_config', 'anz_seed']

    df_temp = pd.DataFrame(dat, columns = columns)
    df = df.append(df_temp)
    example_optim[(opt, trial)] = run.optim_list


# ======================== /
# Save data
# ======================== /
fname = str(anz.__class__).split('.')[2].split("'")[0] + str(cst.__class__).split('cost.')[1].split("'")[0]
fname += '_{}qu'.format(NB_SPINS)
fname = fname + '_{}calls_{}ratio'.format(NB_CALLS,NB_IN_IT_RATIO).replace('.', 'p') + ut.safe_string.gen() + '.pkl'
if SAVE_DATA:
    dict_to_dill = {'df':df,
                    'anz':anz,
                    'example_optim':example_optim,
                    'NB_IN_IT_RATIO':NB_IN_IT_RATIO,
                    'NB_CALLS':NB_CALLS,
                    'NB_SHOTS_DEFAULT':NB_SHOTS_DEFAULT,
                    'hamiltonian':hamiltonian,
                    'NB_ANZ_SEED':NB_ANZ_SEED,
                    'NB_HAM_SEED':NB_HAM_SEED}
    with open(fname, 'wb') as f:
        dill.dump(dict_to_dill, f)





#%% LOAD / PLOT DATA
# ========================= /
# Files:    diff configs  1 trial 4 qubits 1000 calls
#           fname = 'RandomAnsatz_5CONF_RandomXYCost_4qu_1000calls_0p5001024ratioIn6.pkl'
#           fname = 'RandomAnsatz_10CONF_RandomXYCost_4qu_1000calls_0p5001024ratiofHg.pkl'
#           fname = 'RandomAnsatz_15CONF_RandomXYCost_4qu_1000calls_0p5001024ratioWip.pkl'

#           fname = 'RandomAnsatz_15CONF_RandomXYCost_4qu_1500calls_0p5001024ratiocm1.pkl'
#           fname = 'RandomAnsatzRandomXYCost_4qu_1500calls_0p5001024ratioZ0S.pkl''
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
        NB_TRIALS = int(df.trial.max()) + 1
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
    h = df.iloc[ii]['h_config']
    h = df.h_config.unique().tolist().index(h)
    axes[0].errorbar(h + 0.1*t/NB_TRIALS, m, yerr = v, fmt = 'r.', label='bopt')
# axes[0].plot([-0.88223511, -0.85469458, -0.83206065, -0.82616974, -0.83419339, -0.85555687, -0.89058293, -0.92913224, -0.95236189, -0.95812539])
axes[0].set_title('Shot noise ({} shots/circ)'.format(NB_SHOTS_DEFAULT))
axes[0].set_ylabel('Cost ' + fname)
axes[0].set_xlabel('h_config')
# axes[0].set_ylim(-0.96, -0.6)



sns.pointplot(data = df, x = 'h_config', y = 'mean', join=False, ax=axes[1])
axes[1].set_title('Optimiser noise')
axes[1].set_xlabel('h_config')
axes[1].set_ylabel('cost')


sns.boxplot(data = df, x = 'h_config', y = 'mean', ax = axes[2])
axes[2].set_title('Optimiser noise')
axes[2].set_xlabel('h_config')
axes[2].set_ylabel('cost')
f.show()




f = plt.figure(2, figsize=(5, 11))
axes = f.subplots(len(df.h_config.unique()), 1, sharex=True)
axes = np.atleast_1d(axes)
for ii, h_config in enumerate(df.h_config.unique()):
    axes[ii].set_title('h_config: {}'.format(h_config))

    axes[ii].set_ylabel('Cost')
    for trial in range(NB_TRIALS):
        bo = example_optim[(1, trial)][ii]
        data = np.ravel(bo.optimiser.Y)
        iter_data = data
        sns.scatterplot(np.arange(1,len(iter_data)+1),  iter_data, ax=axes[ii])
axes[ii].set_xlabel('iter')
f.show()




f = plt.figure(3, figsize=(10, 5))
axes = f.subplots(1, len(df.h_config.unique()), sharey=True, squeeze=False)
axes = np.ravel(axes)
for opt, trial in example_optim.keys():
    for ii, h_config in enumerate(df.h_config.unique()):
        bo = example_optim[(opt,trial)][ii].optimiser
        x = ut._diff_between_x(bo.X)
        sns.lineplot(np.arange(1, len(x) + 1),  x, ax=axes[ii])
        axes[ii].set_title('h_config: {}'.format(h_config))
        axes[ii].set_xlabel('iter')
f.show()
axes[0].set_ylabel('x_{i + 1} - x_{i}')
[ax.set_xlabel('iter') for ax in axes]
f.show()


# f = plt.figure(10)
# axes = f.subplots(int(np.sqrt(len(hvec))), int(np.sqrt(len(hvec))), sharey=True, sharex = True, squeeze=False)
# axes = np.ravel(axes)
# for ii, h in enumerate(hvec):
#     pt = int(np.sqrt(len(hvec)))**2
#     axes[ii%pt].pcolor(np.log(h))


# def check_ham_configs(hvec):
#     for ct, h in enumerate(hvec):
#         for rr, dat in enumerate(h):
#             plt.plot(dat[(rr+1):], '.-', label = str(ct))
# check_ham_configs(hvec)


