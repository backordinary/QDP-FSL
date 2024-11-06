# https://github.com/achieveordie/HybridQNN/blob/9bc236216cfcf6eaff7c2d7becbcc930d161c7ba/main.py
import numpy as np
from torch.autograd import Function
from torchvision import datasets, transforms
import torch.optim as optim
import torch
import torch.nn.functional as F

import qiskit

# import from ./circuits/
from circuits.circuit_ry import Hybrid


# Define the custom autograd Function
class HybridFunction(Function):
    """ This class is uses `parameter shift rule` to calculate the gradients
        and inherits the pytorch's `Function` from `autograd` to have two methods
        `forward` and `backward` to calculate the gradients"""

    @staticmethod
    def forward(ctx, input, quantum_circuit, shift):
        ctx.shift = shift
        ctx.quantum_circuit = quantum_circuit
        #expectations = ctx.quantum_circuit.run(input.tolist())
        expectations = ctx.quantum_circuit.run(input[0].tolist())
        ctx.save_for_backward(input, torch.tensor([expectations]))
        return torch.tensor([expectations])

    @staticmethod
    def backward(ctx, grad_output):
        input, expectations = ctx.saved_tensors
        input_list = np.array(input.tolist())
        shift_right = input_list + np.ones(input_list.shape) * ctx.shift
        shift_left = input_list - np.ones(input_list.shape) * ctx.shift

        gradients = []
        for i in range(len(input_list)):
            expectation_right = ctx.quantum_circuit.run(shift_right[i])
            expectation_left = ctx.quantum_circuit.run(shift_left[i])
            gradient = torch.tensor([expectation_right]) - torch.tensor([expectation_left])
            gradients.append(gradient)
        gradients = np.array([gradients]).T
        #print(gradients, grad_output)
        return torch.tensor([gradients]).float() * grad_output.float(), None, None
        # three values returned from backward because 3 values are passed to forward
   

""" Now preprocess data from MNIST datasets"""
# For train data:
n_samples = 100
X_train = datasets.MNIST(root='./data', download=False, train=True, transform=
                         transforms.Compose([transforms.ToTensor()]))
idx = np.append(np.where(X_train.targets == 0)[0][:n_samples],
                np.where(X_train.targets == 1)[0][:n_samples]) # we shall only consider binary classification

X_train.data = X_train.data[idx]
X_train.targets = X_train.targets[idx]

trainloader = torch.utils.data.DataLoader(X_train, batch_size=1, shuffle=True)
# For test data
n_samples = 50
X_test = datasets.MNIST(root='./data', download=False, train=False, transform=
                        transforms.Compose([transforms.ToTensor()]))
idx = np.append(np.where(X_test.targets == 0)[0][:n_samples],
                np.where(X_test.targets == 1)[0][:n_samples])

X_test.data = X_test.data[idx]
X_test.targets = X_test.targets[idx]

testloader = torch.utils.data.DataLoader(X_test, batch_size=1, shuffle=False)


# Now is the time to define the Hybrid Architecture
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(6, 16, kernel_size=5)
        self.dropout = torch.nn.Dropout2d()
        self.fc1 = torch.nn.Linear(256, 64)
        self.fc2 = torch.nn.Linear(64, 1)
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


# Now to train the above Hybrid Model
model = Net()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_func = torch.nn.NLLLoss()

epochs = 20
loss_list = []

model.train()
for epoch in range(epochs):
    total_loss = []
    for batch_idx, (data, target) in enumerate(trainloader):
        optimizer.zero_grad()
        output = model(data)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        total_loss.append(loss.item())
    loss_list.append(sum(total_loss) / len(total_loss))
    print('Training [{:.0f}%]\tLoss: {:.4f}'.format(
        100. * (epoch + 1) / epochs, loss_list[-1]))

# Test the data on test_loader
model.eval()
with torch.no_grad():
    correct = 0
    for batch_idx, (data, target) in enumerate(testloader):
        output = model(data)

        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

        loss = loss_func(output, target)
        total_loss.append(loss.item())

    print('Performance on test data:\n\tLoss: {:.4f}\n\tAccuracy: {:.1f}%'.format(
        sum(total_loss) / len(total_loss),
        correct / len(testloader) * 100)
    )
