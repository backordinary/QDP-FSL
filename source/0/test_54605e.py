# https://github.com/boyesjo/tma4500/blob/41c56c2cd29e3763832334701fd2c676f2cb4bc8/code/conv_mnist/test.py
# %% https://medium.com/codex/hybrid-quantum-classical-neural-network-for-classification-of-images-in-fashionmnist-dataset-7274364f7dcd # noqa
# https://github.com/Q-MAB/Qiskit-FashionMNIST-Case/blob/main/FashionMNIST%20case%20study.py # noqa
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from qiskit import Aer
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.opflow import AerPauliExpectation
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.fake_provider import FakeMontreal as FakeBackend
from qiskit.utils import QuantumInstance
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import TwoLayerQNN
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# %%
qi = QuantumInstance(Aer.get_backend("qasm_simulator"), shots=1024)
qi_noisy = QuantumInstance(
    Aer.get_backend("qasm_simulator"),
    shots=1024,
    noise_model=NoiseModel.from_backend(FakeBackend()),
)

# %%
training_data = datasets.MNIST(
    download=True,
    root="data",
    train=True,
    transform=transforms.ToTensor(),
)

test_data = datasets.MNIST(
    download=True,
    root="data",
    train=False,
    transform=transforms.ToTensor(),
)


# only have targets 0  and 1
def filter_data(data):
    idx = np.where((data.targets == 0) | (data.targets == 1))[0]
    data.data = data.data[idx]
    data.targets = data.targets[idx]


filter_data(training_data)
filter_data(test_data)

num_samples = 512
batch_size = 64

training_data.data = training_data.data[:num_samples]
training_data.targets = training_data.targets[:num_samples]
test_data.data = test_data.data[:num_samples]
test_data.targets = test_data.targets[:num_samples]

train_loader = DataLoader(
    training_data,
    batch_size=batch_size,
    shuffle=True,
)

test_loader = DataLoader(
    test_data,
    batch_size=batch_size,
    shuffle=True,
)


# %%
# plots six first images in a 2x3 grid
fig, axes = plt.subplots(2, 3)
for i, ax in enumerate(axes.flatten()):
    ax.imshow(training_data[i][0].squeeze(), cmap="gray_r")
    ax.set_title(training_data[i][1])
    ax.axis("off")

# %%
feature_map = ZZFeatureMap(feature_dimension=2, entanglement="linear")
ansatz = RealAmplitudes(2, reps=1, entanglement="linear")
qnn = TwoLayerQNN(
    2,
    feature_map,
    ansatz,
    input_gradients=True,
    exp_val=AerPauliExpectation(),
    quantum_instance=qi,
)

qnn_noisy = TwoLayerQNN(
    2,
    feature_map,
    ansatz,
    input_gradients=True,
    exp_val=AerPauliExpectation(),
    quantum_instance=qi_noisy,
)


# %%
class QuantumNet(nn.Module):
    def __init__(self, qnn):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 2, kernel_size=5)
        self.conv2 = nn.Conv2d(2, 16, kernel_size=5)
        self.dropout = nn.Dropout2d()
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 2)
        self.qnn = TorchConnector(qnn)
        self.fc3 = nn.Linear(1, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.qnn(x)
        x = self.fc3(x)
        return torch.cat((x, 1 - x), -1)


class ClassicalNet(nn.Module):
    # same as QuantumNet but without the qnn
    # instead 4 classical parameters are used
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 2, kernel_size=5)
        self.conv2 = nn.Conv2d(2, 16, kernel_size=5)
        self.dropout = nn.Dropout2d()
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 2)
        self.fc3 = nn.Linear(2, 2, bias=True)
        self.fc4 = nn.Linear(2, 1, bias=True)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.fc4(x)
        return torch.cat((x, 1 - x), -1)


# %%
def train(model: nn.Module, epochs: int = 20) -> np.ndarray:

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_func = nn.NLLLoss()
    loss_list = np.zeros((epochs, len(train_loader)))

    for epoch in range(epochs):

        for batch, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = loss_func(output, target)
            loss.backward()
            optimizer.step()
            loss_list[epoch, batch] = loss.item()

        print(
            f"Epoch: {epoch} ",
            f"Loss: {loss_list[epoch].mean()} +- {loss_list[epoch].std()}",
        )

    return np.array(loss_list)


# %%
model_qnn = QuantumNet(qnn)
loss_qnn = train(model_qnn)

# %%
model_qnn_noisy = QuantumNet(qnn_noisy)
loss_qnn_noisy = train(model_qnn_noisy)

# %%
model_classical = ClassicalNet()
loss_classical = train(model_classical, epochs=100)


# %%plot all losses with std dev
def plot_loss(loss_list, label, color):
    mean = loss_list.mean(axis=1)
    std = loss_list.std(axis=1)
    plt.plot(mean, label=label, color=color)
    plt.fill_between(
        np.arange(len(mean)),
        mean - std,
        mean + std,
        alpha=0.2,
        color=color,
    )


# plot_loss(loss_qnn, "QNN", "blue")
plot_loss(loss_classical, "Classical", "red")
plt.legend()
plt.show()
# %%
torch.save(model_qnn.state_dict(), "qnn.pt")


# %%
# test the model
def test(model: nn.Module) -> float:
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    return correct / total


print(test(model_qnn))
print(test(model_classical))

# %%
