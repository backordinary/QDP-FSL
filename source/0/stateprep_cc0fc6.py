# https://github.com/andrewliu2001/quantum-ensemble/blob/136b5eb00fce5ce79c69a52fe8f47a9e9a05e780/stateprep.py
import qiskit
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d
import mpl_toolkits.mplot3d.art3d as art3d
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.visualization import plot_bloch_multivector




def stateprep(x_train, y_train, x_test, d):

  """
  x_train: array of training features (n_samples, 2)
  y_train: array of binary training labels (n_samples, )
  x_test: array of test features (2, )
  d: number of control qubits. Generates 2^d transformations to training data
  """

  N = x_train.shape[0]

  control_reg = QuantumRegister(max(d, 1), 'control')
    
  x_train_reg = QuantumRegister(N, 'x_train')
  y_train_reg = QuantumRegister(N, 'y_train')
  x_test_reg = QuantumRegister(1, 'x_test')
  prediction_reg = QuantumRegister(1, 'prediction')
  cr = ClassicalRegister(1, name = "cr")



  stateprep = QuantumCircuit(control_reg, x_train_reg, y_train_reg, x_test_reg, prediction_reg, cr)

  #create uniform superposition of control qubits
  for i in range(d):
    stateprep.h(control_reg[i])


  #initialize training data
  for i in range(x_train.shape[0]):
    stateprep.initialize(x_train[i]/np.linalg.norm(x_train[i]), i+d)

  for i in range(y_train.shape[0]):
    if y_train[i] == 1:
      stateprep.initialize([0, 1], i+d+x_train.shape[0])
    else:
      stateprep.initialize([1, 0], i+d+x_train.shape[0])

  #initialize test data
  stateprep.initialize(x_test/np.linalg.norm(x_test), d+2*N)


  stateprep.barrier()

  return stateprep