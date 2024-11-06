# https://github.com/constatza/qlib/blob/a0d89ce4a227176f21c4fc75cf95cfaf2d2a670b/experiments/custom4x4/experiment-surrogate.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 10:25:47 2022

@author: archer
"""

import os
import numpy as np
from keras.models import load_model
from qiskit import Aer
from scipy.optimize import basinhopping
from qiskit.circuit.library import RealAmplitudes

from qlib.utils import states2qubits, FileLogger
from qlib.solvers.vqls import VQLS, FixedAnsatz, Experiment, RealAmplitudesAnsatz, \
    SolutionPredictorSurrogate


initial_parameter_provider = 'mlp'
input_path_physical_parameters = os.path.join("input", "parameters.in")

parameters = np.loadtxt(input_path_physical_parameters)

x = parameters[:, 0]
y = parameters[:, 1]

matrices = np.array([[-0.5*x**2, x*y],
                     [x*y, 2*y**2 + 1]])

matrices = matrices.transpose(2, 0, 1)

matrices = np.block([[3*matrices, 2*matrices],
                     [2*matrices, 4*matrices]])

N = matrices.shape[2]
rhs = np.zeros((N,))
rhs[0] = 1

num_qubits = states2qubits(N)
num_samples = matrices.shape[0]


ansatz = FixedAnsatz(num_qubits=num_qubits, num_layers=1,
                     max_parameters=3)

optimizer = 'bfgs'
optimization_options = {'tol': 1e-4}

model = load_model('model0')

if initial_parameter_provider == 'mlp':
    predictor = SolutionPredictorSurrogate(
        model, parameters, training_size=int(0.2*num_samples))
elif initial_parameter_provider is None:
    predictor = None

experiment = Experiment(matrices, rhs, ansatz,
                        optimizer=optimizer,
                        initial_parameter_predictor=predictor,
                        save_path='output/'+ initial_parameter_provider,
                        dateit=True,
)


experiment.run(**optimization_options)
