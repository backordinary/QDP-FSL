# https://github.com/andrewliu2001/quantum-ensemble/blob/136b5eb00fce5ce79c69a52fe8f47a9e9a05e780/sampler.py
import qiskit
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d
import mpl_toolkits.mplot3d.art3d as art3d
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.visualization import plot_bloch_multivector
import random

def superposition_sampler(N, d):

    """
    x_train: array of training features
    y_train: array of binary training labels
    x_test: array of test features
    d: number of control qubits. Generates 2^d transformations to training data
    """

    control_reg = QuantumRegister(max(d,1), 'control')
    x_train_reg = QuantumRegister(N, 'x_train')
    y_train_reg = QuantumRegister(N, 'y_train')
    x_test_reg = QuantumRegister(1, 'x_test')
    prediction_reg = QuantumRegister(1, 'prediction')
    cr = ClassicalRegister(1, name = "cr")


    sampler = QuantumCircuit(control_reg, x_train_reg, y_train_reg, x_test_reg, prediction_reg, cr)


    for i in range(d):
        l, m = random.sample(range(0,N), 2)
        lp, mp = random.sample(range(0,N), 2)

        sampler.cswap(control_reg[i], x_train_reg[l], x_train_reg[m])
        sampler.cswap(control_reg[i], y_train_reg[l], y_train_reg[m])
        sampler.x(i)
        sampler.cswap(control_reg[i], x_train_reg[lp], x_train_reg[mp])
        sampler.cswap(control_reg[i], y_train_reg[lp], y_train_reg[mp])
        sampler.barrier()


    return sampler