# https://github.com/jamendez99/SuperTangle/blob/4ed069529b1536353f3edddc48bd9063ab782133/circuit_vis.py
from qiskit.tools.visualization import circuit_drawer
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

import torch
from torch.autograd import Function
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import qiskit
from qiskit.visualization import *

from QuantumNet import QuantumNet

from sklearn.model_selection import train_test_split
import sklearn

#----------------
# Load the models
#----------------
dir_path = os.path.dirname(os.path.realpath(__file__))
models_path = str(dir_path) + '/res/european/qnn/models/'
vis_path = str(dir_path) + '/res/european/qnn/vis/'

for i in range(1, 15):
    model = QuantumNet(i)
    model.load_state_dict(torch.load(models_path + 'qnn_' + str(i) + '.pt'))
    model.eval()
    circuit_drawer(
        model.hybrid.quantum_circuit._circuit,
        output='mpl', 
        filename=vis_path + 'circuit_' + str(i) + '.pdf')
    