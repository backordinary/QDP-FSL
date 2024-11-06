# https://github.com/Korea-Quantum-Computing/QuantumVariableSelection/blob/707aa2d3655de28778f80782ec795eaa66dfe865/module/kqc_custom.py
# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 16:23:42 2022

@author: RB
"""

import pandas as pd
import numpy as np
from qiskit_optimization import QuadraticProgram
from qiskit import Aer
from qiskit.utils import algorithm_globals, QuantumInstance
from qiskit.algorithms import QAOA, NumPyMinimumEigensolver
from qiskit_optimization.algorithms import (
    MinimumEigenOptimizer,
    RecursiveMinimumEigenOptimizer,
    SolutionSample,
    OptimizationResultStatus,
)
from qiskit_optimization import QuadraticProgram
from qiskit.visualization import plot_histogram
from typing import List, Tuple
from sklearn.utils import check_random_state


__all__ = ["qubo_qaoa","qubo_exact","generate_independent_sample","generate_dependent_sample"]

def qubo_qaoa(Q,beta,backend = Aer.get_backend("qasm_simulator")):
    algorithm_globals.massive = True
    p = Q.shape[0]
    mod = QuadraticProgram("my problem")
    linear = {"x"+str(i): beta[i] for i in range(p)}
    quadratic = {("x"+str(i),"x"+str(j)): Q.values[i,j] for i in range(p) for j in range(p)}

    for i in range(p) :
        mod.binary_var(name="x"+str(i))

    mod.minimize(linear=linear,quadratic=quadratic)
    quantum_instance = QuantumInstance(backend)
    mes = QAOA(quantum_instance=quantum_instance)
    optimizer = MinimumEigenOptimizer(mes)
    result = optimizer.solve(mod)
    return([result,mod])

def qubo_exact(Q,beta):
    algorithm_globals.massive = True
    p = Q.shape[0]
    mod = QuadraticProgram("my problem")
    linear = {"x"+str(i): beta[i] for i in range(p)}
    quadratic = {("x"+str(i),"x"+str(j)): Q.values[i,j] for i in range(p) for j in range(p)}

    for i in range(p) :
        mod.binary_var(name="x"+str(i))

    mod.minimize(linear=linear,quadratic=quadratic)
    mes = NumPyMinimumEigensolver()
    optimizer = MinimumEigenOptimizer(mes)
    result = optimizer.solve(mod)
    return([result,mod])


def generate_independent_sample(n_samples=500, n_features=10, beta_coef =[4,3,2,2],epsilon=4, random_state=None):

        rng = check_random_state(random_state)
        if n_features < 4:
            raise ValueError("`n_features` must be >= 4. "
                             "Got n_features={0}".format(n_features))
        # normally distributed features
        X = rng.randn(n_samples, n_features)
        # beta is a linear combination of informative features
        n_informative = len(beta_coef)
        beta = np.hstack((
            beta_coef, np.zeros(n_features - n_informative)))
        # cubic in subspace
        y = np.dot(X, beta)
        y += epsilon * rng.randn(n_samples)
        return X, y

def generate_dependent_sample(n_samples=500, n_features=10, beta_coef =[4,3,2,2],epsilon=4,covariance_parameter=1, random_state=None):

        rng = check_random_state(random_state)
        if n_features < 4:
            raise ValueError("`n_features` must be >= 4. "
                             "Got n_features={0}".format(n_features))

        # normally distributed features

        v = rng.normal(0, 0.4, (n_features, n_features))
        mean = np.zeros(n_features)
        cov = v @ v.T*covariance_parameter + 0.1 * np.identity(n_features)
        X = rng.multivariate_normal(mean, cov, n_samples)

        # beta is a linear combination of informative features
        n_informative = len(beta_coef)
        beta = np.hstack((
            beta_coef, np.zeros(n_features - n_informative)))

        # cubic in subspace
        y = np.dot(X, beta)
        y += epsilon * rng.randn(n_samples)

        return X, y