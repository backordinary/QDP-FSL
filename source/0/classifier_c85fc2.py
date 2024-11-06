# https://github.com/andrewliu2001/quantum-ensemble/blob/136b5eb00fce5ce79c69a52fe8f47a9e9a05e780/classifier.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 23:20:27 2022

@author: erinnsun
"""

import qiskit
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d
import mpl_toolkits.mplot3d.art3d as art3d
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.visualization import plot_bloch_multivector
import random



def cosine_classifier(N, d):

  """
  N: number of train samples
  d: number of control qubits. Generates 2^d transformations to training data
  """
  control_reg = QuantumRegister(max(d,1), 'control')
  x_train_reg = QuantumRegister(N, 'x_train')
  y_train_reg = QuantumRegister(N, 'y_train')
  x_test_reg = QuantumRegister(1, 'x_test')
  prediction_reg = QuantumRegister(1, 'prediction')
  cr = ClassicalRegister(1, name = "cr")



  f = QuantumCircuit(control_reg, x_train_reg, y_train_reg, x_test_reg, prediction_reg, cr)
  f.h(prediction_reg[0])

  k = random.sample(range(0, N), 1)

  f.cswap(prediction_reg[0], x_train_reg[k], x_test_reg[0])
  f.h(prediction_reg[0])
  f.cx(y_train_reg[k], prediction_reg[0])
  f.barrier()
  
  return f


def measure(N, d):
    """
    N: number of train samples
    d: number of control qubits. Generates 2^d transformations to training data
    """
        
    control_reg = QuantumRegister(max(d,1), 'control')
    x_train_reg = QuantumRegister(N, 'x_train')
    y_train_reg = QuantumRegister(N, 'y_train')
    x_test_reg = QuantumRegister(1, 'x_test')
    prediction_reg = QuantumRegister(1, 'prediction')
    cr = ClassicalRegister(1, name = "cr")
    
    m = QuantumCircuit(control_reg, x_train_reg, y_train_reg, x_test_reg, prediction_reg, cr)
    m.measure(prediction_reg[0], cr[0])
    
    return m
