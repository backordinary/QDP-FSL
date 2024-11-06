# https://github.com/mrdobson/SENSIP_2021_REU/blob/4fdbfb62497db3060560e06d0988ddd7b2a41308/python/mnist_qnn_prototype.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
##############################################################################
# QNN Operating on MNIST data set (only two labels at a time)
# Matthew Dobson
# github - 
# 
##############################################################################

import numpy as np
import matplotlib.pyplot as plt
import datetime

from torch import tensor
from torch import cat, no_grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.nn import (Module, Conv2d, Linear, Dropout2d, NLLLoss, 
                      MaxPool2d, Flatten, Sequential, ReLU)
import torch.optim as optim
import torch.nn.functional as F

from qiskit import Aer, QuantumCircuit
from qiskit.opflow import Z, I, StateFn, PauliSumOp, AerPauliExpectation, ListOp, Gradient
from qiskit.utils import QuantumInstance
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.algorithms.optimizers import COBYLA, L_BFGS_B, ADAM

from qiskit_machine_learning.neural_networks import TwoLayerQNN, CircuitQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier, VQC
from qiskit_machine_learning.algorithms.regressors import NeuralNetworkRegressor, VQR

from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.exceptions import QiskitMachineLearningError
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import confusion_matrix
# ---------------
# mode parameters
# ---------------
qi = QuantumInstance(Aer.get_backend('qasm_simulator'), shots=1024)
#qi = QuantumInstance(Aer.get_backend('statevector_simulator'))

# modify these parameters to change which two characters we're going to
# compare against in the model, want to expand to 3 and then on
samp_val_1 = 1
samp_val_2 = 7

# init model runtime
begin_time = datetime.datetime.now()

#-------------------------------------------
# MNIST Data Load
#-------------------------------------------
# training set
n_samples = 210

# returns tuple with X_train.data and X_train.targets <-- what I don't have from iris
X_train = datasets.MNIST(root='./data', train=True, download=True,
                         transform=transforms.Compose([transforms.ToTensor()]))

# leave only labels 0 and 1 first
idx = np.append(np.where(X_train.targets == samp_val_1)[0][:n_samples],
                np.where(X_train.targets == samp_val_2)[0][:n_samples])

X_train.data = X_train.data[idx]
X_train.targets = X_train.targets[idx]
# samp vals tunable above in the settings, this is to fit to our data loader
X_train.targets[X_train.targets==samp_val_1] = 0
X_train.targets[X_train.targets==samp_val_2] = 1
### DEBUG    
#print("xtrain targets ", X_train.targets, "\n")

# perform training load
train_loader = DataLoader(X_train, batch_size=1, shuffle=True)

# testing set
n_samples = 90

X_test = datasets.MNIST(root='./data', train=False, download=True,
                        transform=transforms.Compose([transforms.ToTensor()]))
# selecting which samples to keep (first 100 0s and 1s)
idx = np.append(np.where(X_test.targets == samp_val_1)[0][:n_samples], 
                np.where(X_test.targets == samp_val_2)[0][:n_samples])

X_test.data = X_test.data[idx]
X_test.targets = X_test.targets[idx]
# samp vals tunable above in the settings, to fit to data loader
X_test.targets[X_test.targets==samp_val_1] = 0
X_test.targets[X_test.targets==samp_val_2] = 1
### DEBUG    
#print("xtest targets ", X_train.targets, "\n")

# perform testing load
test_loader = DataLoader(X_test, batch_size=1, shuffle=True)

#------------   
# define QNN
#------------
num_inputs = 2

# ZZ is 2nd order Pauli expansion circuit
fm = ZZFeatureMap(num_inputs)
#fm.draw(output='mpl')

# RealAmplitudes is used as an ansatz for ML, heuristic trial wave func
ansatz = RealAmplitudes(num_inputs, reps=1)
#ansatz.draw(output='mpl')

# define observable
observable = PauliSumOp.from_list([('Z'*num_inputs, 1)])
print(observable)

# define two layer QNN
qnn = TwoLayerQNN(num_inputs, 
                  feature_map=fm, 
                  ansatz=ansatz, 
                  observable=observable,
                  quantum_instance=qi)
print(qnn.operator)

class Net(Module):

    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(1, 2, kernel_size=5)
        self.conv2 = Conv2d(2, 16, kernel_size=5)
        self.dropout = Dropout2d()
        self.fc1 = Linear(256, 64)
        self.fc2 = Linear(64, 2)         # 2-dimensional input to QNN
        self.qnn = TorchConnector(qnn)  #
        self.fc3 = Linear(1, 1)          # 1-dimensional output from QNN

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        x = x.view(1, -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.qnn(x)  # apply QNN
        x = self.fc3(x)
        return cat((x, 1 - x), -1)

model = Net()

# define model, optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_func = NLLLoss()
#----------------
# start training
#----------------
epochs = 20    # set num epochs
loss_list = [] # store loss history
model.train()  # place model in training mode
for epoch in range(epochs):
    total_loss = []
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad(set_to_none=True) # init gradient
        output = model(data)             # forward pass
        ### DEBUG
        #print("target is: ", target)
        #print("output is: ", output)
        loss = loss_func(output, target) # calc loss
        loss.backward()                  # backward pass
        optimizer.step()                 # optimize weights
        total_loss.append(loss.item())   # store loss
    loss_list.append(sum(total_loss)/len(total_loss))
    print('Training [{:0f}%]\tLoss: {:.4f}'.format(
           100. * (epoch + 1) / epochs, loss_list[-1]))

plt.figure(3)
plt.plot(loss_list)
plt.title('Hybrid NN Training Convergence')
plt.xlabel('Training Iterations')
plt.ylabel('Neg. Log Likelihood Loss')
plt.show()

#--------------------
# perform evaluations
#--------------------
model.eval() # set into eval mode
with no_grad():
    correct = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        output = model(data)
        if len(output.shape) == 1:
            output = output.reshape(1, *output.shape)
            
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        
        loss = loss_func(output, target)
        total_loss.append(loss.item())
    # batch_size goes where the 1 is here
    print('Performance on test data:\n\tLoss: {:.4f}\n\tAccuracy: {:.1f}%'
           .format(sum(total_loss) / len(total_loss),
                correct / len(test_loader) / 1 * 100))
 

# calculate program runtime
end_time = datetime.datetime.now()
print("\n\nModel Runtime is: ", (end_time - begin_time))
