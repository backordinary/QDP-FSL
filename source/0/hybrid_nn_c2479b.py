# https://github.com/calwoo/qlearn/blob/8ee23033baf69091c944132378f718ea6c4ce950/ml/hybrid_nn.py
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.functional as F
from torchvision import datasets, transforms

import qiskit
from qiskit import assemble


class QuantumCircuit:
    def __init__(self, n_qubits, backend, shots):
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        qbits = list(range(n_qubits))
        self.theta = qiskit.circuit.Parameter("theta")

        self._circuit.h(qbits)
        self._circuit.barrier()
        self._circuit.ry(self.theta, qbits)

        self._circuit.measure_all()

        self.backend = backend
        self.shots = shots

    def run(self, thetas):
        transpiled_qc = qiskit.transpile(self._circuit, self.backend)
        
        # perform measurement
        qobj = assemble(transpiled_qc,
                        shots=self.shots,
                        parameter_binds=[{self.theta: theta} for theta in thetas])

        job = self.backend.run(qobj)
        result = job.result().get_counts()

        # compute expectation (qiskit isn't as cool as pennylane it seems)
        counts = np.array(list(result.values()))
        states = np.array(list(result.keys())).astype(np.float32)
        probs = counts / self.shots

        pauli_z_expectation = np.sum(states * probs)
        return np.array([pauli_z_expectation])


sim = qiskit.Aer.get_backend("qasm_simulator")
circuit = QuantumCircuit(1, sim, 100)
circuit._circuit.draw(output="mpl")
plt.show()


class HybridClassicalQuantumFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, qcircuit, shift):
        ctx.shift = shift
        ctx.qcircuit = qcircuit

        # wrap the expectation evaluation of quantum circuit in torch
        pauli_z_expectation = ctx.qcircuit.run(inp[0].tolist())
        result = torch.tensor([pauli_z_expectation])

        ctx.save_for_backward(inp, result)
        return result

    @staticmethod
    def backward(ctx, grad_out):
        inp, result = ctx.saved_tensors
        inp_list = np.array(inp.tolist())

        # parameter-shift rule:
        #   d_\theta f(x) = f(x + shift) - f(x - shift)
        shift_right = inp_list + np.ones(inp_list.shape) * ctx.shift
        shift_left = inp_list - np.ones(inp_list.shape) * ctx.shift

        grads = []
        for i in range(len(inp_list)):
            expectation_right = ctx.qcircuit.run(shift_right[i])
            expectation_left = ctx.qcircuit.run(shift_left[i])

            grad = expectation_right - expectation_left
            grads.append(grad)
        grads = np.array([grads]).T
        return torch.tensor([grads]).float() * grad_out.float(), None, None


class HybridQuantumNN(nn.Module):
    def __init__(self, backend, shots, shift):
        super(HybridQuantumNN, self).__init__()
        self.qcircuit = QuantumCircuit(1, backend, shots)
        self.shift = shift

    def forward(self, inp):
        return HybridClassicalQuantumFunction.apply(inp, self.qcircuit, self.shift)


# data
# train
n_samples = 100
X_train = datasets.MNIST(root="./data", train=True, download=True,
                         transform=transforms.Compose([transforms.ToTensor()]))

# Leaving only labels 0 and 1 
idx = np.append(np.where(X_train.targets == 0)[0][:n_samples], 
                np.where(X_train.targets == 1)[0][:n_samples])

X_train.data = X_train.data[idx]
X_train.targets = X_train.targets[idx]

train_loader = torch.utils.data.DataLoader(X_train, batch_size=1, shuffle=True)

# test
n_samples = 50

X_test = datasets.MNIST(root="./data", train=False, download=True,
                        transform=transforms.Compose([transforms.ToTensor()]))

idx = np.append(np.where(X_test.targets == 0)[0][:n_samples], 
                np.where(X_test.targets == 1)[0][:n_samples])

X_test.data = X_test.data[idx]
X_test.targets = X_test.targets[idx]

test_loader = torch.utils.data.DataLoader(X_test, batch_size=1, shuffle=True)


# model
class NNet(nn.Module):
    def __init__(self):
        super(NNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.dropout = nn.Dropout2d()
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 1)
        self.sim = qiskit.Aer.get_backend("qasm_simulator")
        self.hybrid_qnn = HybridQuantumNN(self.sim, 100, np.pi / 2)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = self.dropout(x)
        x = x.view(1, -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.hybrid_qnn(x)
        return torch.cat((x, 1-x), -1)


# training
model = NNet()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.NLLLoss()

epochs = 20
losses = []
for epoch in range(epochs):
    epoch_losses = []
    for i, (data, tgt) in enumerate(train_loader):
        out = model(data)
        loss = loss_fn(out, tgt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())

    losses.append(sum(epoch_losses) / len(epoch_losses))
    print("training [{:.0f}%]\tLoss: {:.4f}".format(100.0*(epoch+1) / epochs, losses[-1]))

plt.plot(losses)
plt.title("Hybrid NN training")
plt.xlabel("iterations")
plt.ylabel("nll loss")
plt.show()

# testing
model.eval()
with torch.no_grad():
    correct = 0
    for i, (data, tgt) in enumerate(test_loader):
        out = model(data)
        pred = out.argmax(dim=1, keepdim=True) 
        correct += pred.eq(tgt.view_as(pred)).sum().item()
        
        loss = loss_fn(out, tgt)
        losses.append(loss.item())
        
    print("performance on test data:\n\tLoss: {:.4f}\n\tAccuracy: {:.1f}%".format(
        sum(losses) / len(losses),
        correct / len(test_loader) * 100))
