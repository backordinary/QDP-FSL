# https://github.com/Irnamosa/QuantumImage/blob/311bdd2fab746e743be9dfd9a063c5176973501e/kernel/featuremap/circuits.py
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, execute
from qiskit import IBMQ, Aer

from qiskit.visualization import plot_state_city

from qiskit.opflow import (StateFn, Zero, One, Plus, Minus, H,
                           DictStateFn, VectorStateFn, CircuitStateFn, OperatorStateFn)
from qiskit.opflow import I, X, Y, Z, CZ
from qiskit.opflow import MatrixOp, CircuitOp

from qiskit.algorithms.optimizers import SPSA

from scipy.stats import unitary_group
from scipy.special import expit

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import random

def phi(S, x):
    '''
    Feature embedding for nqubits = 2
    '''
    if len(S)==2:
        return (np.pi - x[0])*(np.pi - x[1])
    else:
        return x[S[0]- 1]
    

def U(x):
    '''
    Unitary of PauliSum operator
    '''
    H = (phi((1,),x)  *  Z^I) + \
        (phi((2,),x)  *  I^Z) + \
        (phi((1,2),x) *  Z^Z)

    return (-1*H).exp_i()

def fancyU(x):
    '''
    Feature map
    '''
    u = U(x)
    return u @ (H^H) @ u @ (H^H) 
