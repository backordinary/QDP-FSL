# https://github.com/Abduhu/vqe/blob/d8e865b1c40e571d40986294636942419673c8b4/vqe.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 10:19:26 2020

@author: abduhu

Application of VQE algorithm to find the ground energy of a given hamiltonian
matrix H with a specifc ansatz and decomposition.

"""

import numpy as np
from copy import deepcopy
from math import pi
from qiskit import(
  QuantumCircuit,
  execute,
  Aer)
from decomposition import (decompose, XX, YY, ZZ)

# Use Aer's qasm_simulator
simulator = Aer.get_backend('qasm_simulator')

# Hamiltonian matrix (H)
H = np.matrix([[1, 0, 0, 0],
               [0, 0, -1, 0],
               [0, -1, 0, 0], 
               [0, 0, 0, 1]])

# Decompose H
decomposition_coefs = decompose(H)

def experiment(theta, n_shots=1000):
    """
    measures the expected value of H applied on the ansatz state with parameter
    angle 'theta'.
    Inputs:
        theta: float:
            angle parameter.
        n_shots: int:
            number of measurments to be performed.
    """
    
    # prepare parametrized state
    circuit = QuantumCircuit(2, 2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.rx(theta, 0)
    
    # measure <ZZ>
    ZZ_VALUES = np.linalg.eig(ZZ)[0]
    circuit_ = deepcopy(circuit)
    circuit_.measure([0,1], [0,1])
    job = execute(circuit_, simulator, shots=n_shots)
    result = job.result()
    counts = result.get_counts(circuit)
    data = []
    if '00' in counts:
        data.append(counts['00'])
    else:
        data.append(0)
    if '01' in counts:
        data.append(counts['01'])
    else:
        data.append(0)
    if '10' in counts:
        data.append(counts['10'])
    else:
        data.append(0)
    if '11' in counts:
        data.append(counts['11'])
    else:
        data.append(0)
    data = np.array(data)
    mean_zz = np.sum(ZZ_VALUES * data) / n_shots
    
    # measure <XX>
    XX_VALUES = np.linalg.eig(XX)[0]
    circuit_ = deepcopy(circuit)
    circuit_.ry(-pi/2, 0)
    circuit_.ry(-pi/2, 1)
    circuit_.measure([0,1], [0,1])
    job = execute(circuit_, simulator, shots=n_shots)
    result = job.result()
    counts = result.get_counts(circuit)
    data = []
    if '00' in counts:
        data.append(counts['00'])
    else:
        data.append(0)
    if '01' in counts:
        data.append(counts['01'])
    else:
        data.append(0)
    if '10' in counts:
        data.append(counts['10'])
    else:
        data.append(0)
    if '11' in counts:
        data.append(counts['11'])
    else:
        data.append(0)
    data = np.array(data)
    mean_xx = np.sum(XX_VALUES * data) / n_shots
    
    # measure <YY>
    YY_VALUES = np.linalg.eig(YY)[0]
    circuit_ = deepcopy(circuit)
    circuit_.rx(pi/2, 0)
    circuit_.rx(pi/2, 1)
    circuit_.measure([0,1], [0,1])
    job = execute(circuit_, simulator, shots=n_shots)
    result = job.result()
    counts = result.get_counts(circuit)
    data = []
    if '00' in counts:
        data.append(counts['00'])
    else:
        data.append(0)
    if '01' in counts:
        data.append(counts['01'])
    else:
        data.append(0)
    if '10' in counts:
        data.append(counts['10'])
    else:
        data.append(0)
    if '11' in counts:
        data.append(counts['11'])
    else:
        data.append(0)
    data = np.array(data)
    mean_yy = np.sum(YY_VALUES * data) / n_shots
    
    
    return np.sum(np.array(decomposition_coefs) * np.array([1, mean_xx,
                                                            mean_yy, mean_zz]))

def find_ground_state(n_points=10, n_shots=1000):
    """
    searchs of the min eigen-value (ground state energy) of the given H matrix.
    Inputs:
        n_points: int:
            number of theta values to try between 0 and 2*pi.
        n_shots: int:
            number of measurments to be performed each time.
    """
    # initiate theta
    new_theta = 0
    ground_state_theta = new_theta
    ground_eigen_value = experiment(new_theta, n_shots).real
    
    # step of theta iteration
    step = 2*pi/n_points
    for nn in range(1, n_points):
        new_theta = nn * step
        new_value = experiment(new_theta, n_shots).real
        if new_value < ground_eigen_value:
            ground_state_theta = new_theta
            ground_eigen_value = new_value
    
    return ground_eigen_value, ground_state_theta
