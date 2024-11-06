# https://github.com/constatza/qlib/blob/a0d89ce4a227176f21c4fc75cf95cfaf2d2a670b/experiments/vqls8x8/experiment1-test.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Sep 29 15:38:15 2022

@author: archer
"""

import numpy as np
from qiskit import Aer
from scipy.io import loadmat
from qlib.solvers.vqls import VQLS, FixedAnsatz, Experiment
from qlib.utils import states2qubits
from qiskit.algorithms.optimizers import POWELL, COBYLA
from qlib.utils import FileLogger

num_layers = 2
num_shots = 2**11
tol = 1e-3

backend = Aer.get_backend('statevector_simulator',
                          max_parallel_threads=0,
                          max_parallel_experiments=100,
                          precision="single")

filepath = "../../data/8x8/"
matrices = loadmat(filepath + "stiffnesses.mat")['stiffnessMatricesData'] \
    .transpose(2, 0, 1).astype(np.float64)
solutions = loadmat(filepath + "solutions.mat")['solutionData']

for matrix in matrices:
    matrix[matrix==2e4] = np.max(matrix)

b = np.zeros((8,))
b[3] = -100
b[6] = 100


size = 4
matrices = np.array(matrices[0:2, :size, :size])
b = np.array([1] + [0]*(size-1))

ansatz = FixedAnsatz(states2qubits(b.shape[0]), num_layers=num_layers)


vqls = VQLS(ansatz=ansatz,
            backend=backend)


powell_options = {'maxfev': 5e3,
                  'ftol': tol}

cobyla_options = {'maxiter': 5e3,
                  'rhobeg': 2,
                  'callback': vqls.print_cost,
                  }

optimizer = POWELL(**powell_options)
optimizer = COBYLA(**cobyla_options, options={'tol':tol})

exp = Experiment(matrices, b,
                 optimizer=optimizer,
                 solver=vqls,
                 backend=backend)

logger = FileLogger([name for name in exp.data.keys()],
                    dateit=True)

exp.run(nearby=True,
        rhobeg=1e-4,
        logger=logger)
