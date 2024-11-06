# https://github.com/jamendez99/SuperTangle/blob/7f1f3dc332a94c603f0cb4ff707979298779352d/regular_results_european.py
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

from RegularNet import RegularNet

from sklearn.model_selection import train_test_split
import sklearn
import csv


# ----------------------------------
# Prepare data from European dataset
# ----------------------------------
dir_path = os.path.dirname(os.path.realpath(__file__))
file_path = str(dir_path) + '/data/european/creditcard.csv'

df = pd.read_csv(file_path)
df = sklearn.utils.shuffle(df)
y = df['Class']
X = df.drop('Class', axis=1)

X_0 = X[y == 0][:1000]
X_1 = X[y == 1]
y_0 = y[y == 0][:1000]
y_1 = y[y == 1]

X = pd.concat([X_0, X_1])
y = pd.concat([y_0, y_1])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# Turn everything to tensors so that Pythorch knows what to do
X_train = torch.from_numpy(X_train.to_numpy()).float()
y_train = torch.from_numpy(y_train.to_numpy())
X_test =  torch.from_numpy(X_test.to_numpy()).float()
y_test =  torch.from_numpy(y_test.to_numpy())

train_ds = torch.utils.data.TensorDataset(X_train, y_train)
test_ds  = torch.utils.data.TensorDataset(X_test , y_test )

batch_size   = 5
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_loader  = torch.utils.data.DataLoader(test_ds , batch_size=batch_size, shuffle=True)

# -------------------------------------------
# Here is where we actually train the network
# -------------------------------------------
loss_path = str(dir_path) + '/res/european/regular/loss.csv'
acc_path = str(dir_path) + '/res/european/regular/accuracy.csv'
models_path = str(dir_path) + '/res/european/regular/models/'

loss_dict = {}
acc_dict  = {}
for quantum_size in range(1, 17):
    model = RegularNet(quantum_size)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_func = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1000/492.0]))

    epochs = 50
    loss_list = []
    acc_list = []
    model.train()
    print("Classical Layer of size {}".format(quantum_size))
    for epoch in range(epochs):
        total_loss = []
        correct = 0
        total_size = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            # Forward pass
            output = model(data).squeeze(1)
            # print(output.shape)
            # exit()
            total_size += list(output.size())[0]
            pred = (output >= 0).float()
            correct += pred.eq(target.view_as(pred).float()).sum().item()
            loss = loss_func(output, target.float())
            # Backward pass
            loss.backward()
            # Optimize the weights
            optimizer.step()
            
            total_loss.append(loss.item())
        loss_list.append(sum(total_loss) / total_size)
        acc_list.append(100 * correct / total_size)
        print('Training [{:.0f}%]\tLoss: {:.4f}\tAccuracy: {:.4f}%'.format(
            100. * (epoch + 1) / epochs, 
            loss_list[-1], 
            acc_list[-1]))
    # loss_dict[quantum_size] = loss_list
    # acc_dict[quantum_size ] = acc_list
    torch.save(model.state_dict(), models_path + 'regular_' + str(quantum_size) + '.pt')
    with open(loss_path, 'a', newline='') as csv_file:  
        writer = csv.writer(csv_file)
        writer.writerow([quantum_size] + loss_list)

    with open(acc_path, 'a', newline='') as csv_file:  
        writer = csv.writer(csv_file)
        writer.writerow([quantum_size] + acc_list)
    # exit()