# https://github.com/Sri-Harsha-T/Quantum-Computing-and-Information/blob/e5471630f6d4a3fdfb4f0fa97085191a6750b1b5/Quantum%20Machine%20Learning%20Algorithms%20with%20Cirq/MNIST/mnist.py
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


class QuantumCircuit:
    """
    The class implements a simple Quantum Block
    """
    def __init__(self, num_qubits, backend, copies: int = 1000):
        self._circuit_ = qiskit.QuantumCircuit(num_qubits)
        self.theta = qiskit.circuit.Parameter('theta')
        self._circuit_.h([i for i in range(num_qubits)])
        self._circuit_.barrier()
        self._circuit_.ry(self.theta, 
        [i for i in range(num_qubits)])
        self._circuit_.measure_all()
        self.backend = backend
        self.copies = copies

    def run(self, theta_batch):
        job = qiskit.execute(self._circuit_,
        self.backend,
        shots=self.copies,
        parameter_binds=[
        {self.theta: theta}
        for theta in theta_batch])
        result = job.result().get_counts(self._circuit_)
        counts = np.array(list(result.values()))
        states = np.array(list(result.keys())).astype(np.float32)
        probs = counts / self.copies
        expectation = np.array([np.sum(np.multiply(probs, states))])
        return expectation

    
class QuantumFunction(Function):
    """ Hybrid quantum - classical function definition """
    @staticmethod
    def forward(ctx, input, q_circuit, shift):
        """ Forward pass computation """
        ctx.shift = shift
        ctx.q_circuit = q_circuit
        theta_batch = input[0].tolist()
        expectation = ctx.q_circuit.run(theta_batch=theta_batch)
        result = torch.tensor([expectation])
        ctx.save_for_backward(input, result)
        return result
    @staticmethod
    def backward(ctx, grad_output):
        """ Backward pass computation """
        input, expectation = ctx.saved_tensors
        theta_batch = np.array(input.tolist())
        shift_right = theta_batch + np.ones(theta_batch.shape) * ctx.shift
        shift_left = theta_batch - np.ones(theta_batch.shape) * ctx.shift
        gradients = []
        for i in range(len(theta_batch)):
            expectation_right = ctx.q_circuit.run(shift_right[i])
            expectation_left = ctx.q_circuit.run(shift_left[i])
            gradient = torch.tensor([expectation_right])
            - torch.tensor([expectation_left])
            gradients.append(gradient)
        gradients = np.array([gradients]).T
        return torch.tensor([gradients]).float()*grad_output.float(), None, None

class QuantumLayer(nn.Module):
    """ Hybrid quantum - classical layer definition """
    def __init__(self,num_qubits, backend, shift, copies=1000):
        super(QuantumLayer, self).__init__()
        self.q_circuit = QuantumCircuit(num_qubits, backend, copies)
        self.shift = shift
    def forward(self, input):
        return QuantumFunction.apply(input,
        self.q_circuit, self.shift)

class QCNNet(nn.Module):
    def __init__(self, num_qubits=1, backend=
    qiskit.Aer.get_backend('qasm_simulator'),
    shift=np.pi/2,
    copies=1000):
        super(QCNNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.dropout = nn.Dropout2d()
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 1)
        self.q_layer = QuantumLayer(num_qubits=num_qubits,
        backend=backend,
        shift=shift,
        copies=copies)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        x = x.view(1, -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.q_layer(x)
        return torch.cat((x, 1 - x), -1)

    # Define the train test data loaders
def train_test_dataloaders(train_samples=1000,
    test_samples=500,
    train_batch_size=1,
    test_batch_size=1):
    X_train = datasets.MNIST(root='./data',
    train=True, download=True,
    transform=transforms.Compose(
    [transforms.ToTensor()]))
    # Extracting only MNIST labels 0 and 1
    idx = np.append(np.where(X_train.targets== 0)[0][:train_samples], np.where(X_train.targets
    == 1)[0][:train_samples])

    X_train.data = X_train.data[idx]
    X_train.targets = X_train.targets[idx]
    train_loader = torch.utils.data.DataLoader(X_train,
    batch_size=train_batch_size, shuffle=True)
    X_test = datasets.MNIST(root='./data',
    train=False, download=True, transform=transforms.Compose(
    [transforms.ToTensor()]))
    idx = np.append(np.where(X_test.targets
    == 0)[0][:test_samples], np.where(X_test.targets
    == 1)[0][:test_samples])
    X_test.data = X_test.data[idx]
    X_test.targets = X_test.targets[idx]
    test_loader = torch.utils.data.DataLoader(X_test,
    batch_size=test_batch_size, shuffle=True)
    return train_loader, test_loader

def main(num_epochs=20,
    lr=.001,
    train_samples=1000,
    test_samples=500,
    train_batch_size=1,
    test_batch_size=1):
    model = QCNNet()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.NLLLoss()
    train_loader, test_loader = train_test_dataloaders(
    train_samples,
    test_samples,
    train_batch_size,
    test_batch_size)
    loss_list = []
    model.train()
    for epoch in range(num_epochs):
        total_loss = []
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            # Take the Forward pass
            output = model(data)
            # Calculate the log loss
            loss = loss_func(output, target)
            # Take the Backward pass
            loss.backward()
            # Update the Model weights
            optimizer.step()
            total_loss.append(loss.item())
        loss_list.append(sum(total_loss) / len(total_loss))
        print('Training [{:.0f}%]\tLoss: {:.4f}'.format(
        100. * (epoch + 1) / num_epochs, loss_list[-1]))
    plt.plot(loss_list)
    plt.title('Hybrid ConvNet Training Convergence')
    plt.xlabel('Training Iterations')
    plt.ylabel('Neg Log Loss')

    model.eval()
    with torch.no_grad():
        correct = 0
        for batch_idx, (data, target) in enumerate(test_loader):
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            loss = loss_func(output, target)
            total_loss.append(loss.item())
        print('Inference on test data:\n\tLoss: {:.4f}\n\tAccuracy: {:.1f}%'.format(
        sum(total_loss) / len(total_loss),
        correct / len(test_loader) * 100)
        )
if __name__ == '__main__':
    main()