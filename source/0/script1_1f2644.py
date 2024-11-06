# https://github.com/claudioalvesmonteiro/QuantumNeuronRealWeights/blob/ab295615a035d2d3b1cf574c64c15cc7c2c89d3f/script1.py
from encodingsource import InitializerUniformlyRotation
from sf import sfGenerator
from hsgs import hsgsGenerator
import numpy as np
import math
import random
from qiskit import execute, Aer, QuantumRegister, QuantumCircuit, ClassicalRegister
from sympy.combinatorics.graycode import GrayCode
from qiskit.aqua.utils.controlled_circuit import apply_cu3
from extrafunctions import *
from neuron import *

np.random.seed(7)

df = pd.read_csv('dataset.csv')
x_train = df.iloc[:, :-1].to_numpy()
y_train = df.iloc[:, -1].to_numpy()

data_len = len(y_train)
input_dim = x_train.shape[-1]
nb_epochs = 20
lr = 0.01
simulator = Aer.get_backend('qasm_simulator')


for threshold in [0.07, 0.08, 0.09, 0.01, 0.001, 0.01, 0.02, 0.1, 0.2,0.3, 0.4,0.5,0.6,0.7]:
    for lr in [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01,0.02,0.03,0.04,0.1, 0.2, 0.3]:
        print("thr", threshold)
        print("lr", lr)
        
        trainNeuron(method='delta',
                    nb_epochs=100,
                    binaryWeights=True,
                    stochastic=False,
                    listOfInput=x_train,
                    listOfExpectedOutput=y_train,
                    circuitGeneratorOfUOperator='hsgs',
                    simulator=simulator,
                    threshold=threshold,
                    lr = lr,
                    memoryOfExecutions={})


# Falta botar output dentro de /testesoutput