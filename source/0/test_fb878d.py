# https://github.com/boyesjo/tma4500/blob/41c56c2cd29e3763832334701fd2c676f2cb4bc8/code/iris_multiclass/test.py
# %%
import numpy as np
import pandas as pd
from qiskit import Aer, QuantumCircuit
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.utils import QuantumInstance
from qiskit_machine_learning import neural_networks
from qiskit_machine_learning.connectors import TorchConnector
from sklearn.datasets import load_iris
from sklearn.preprocessing import normalize
from torch import Tensor, nn, optim, tensor
from torch.nn import functional as F

NUM_QUBITS = 4


# %%
def load_data() -> tuple[Tensor, Tensor]:
    iris = load_iris()
    df = pd.DataFrame(
        data=np.c_[iris["data"], iris["target"]],
        columns=iris["feature_names"] + ["target"],
    )

    x = normalize(df.drop("target", axis=1).values)
    y = df["target"].values.astype(int)

    # shuffle
    idx = np.random.permutation(len(x))
    x = x[idx]
    y = y[idx]

    return tensor(x).float(), tensor(y).long()


x, y = load_data()


def interpret(i: int) -> tuple[int, int, int]:
    ret = (i & 1, (i >> 1) & 1, (i >> 2) & 1)
    return ret


def create_qnn():
    qi = QuantumInstance(backend=Aer.get_backend("aer_simulator"))

    fm = ZZFeatureMap(feature_dimension=NUM_QUBITS, reps=2)

    ansatz = RealAmplitudes(num_qubits=NUM_QUBITS, reps=1)

    qc = QuantumCircuit(NUM_QUBITS)
    qc.append(fm, range(NUM_QUBITS))
    qc.append(ansatz, range(NUM_QUBITS))

    qnn = neural_networks.CircuitQNN(
        qc,
        input_params=fm.parameters,
        weight_params=ansatz.parameters,
        interpret=interpret,
        # output_shape=(2, 2, 2),
        sampling=100,
        quantum_instance=qi,
        input_gradients=True,
    )

    return qnn


class QNN(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.qnn = TorchConnector(create_qnn(**kwargs))

    def forward(self, x):
        x = self.qnn.forward(x)
        x = x.sum(dim=0)
        x = F.softmax(x, dim=-1)
        return x


QNN()(x[0])


# %%
class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 1, bias=True)
        self.fc2 = nn.Linear(1, 3, bias=False)

    def forward(self, x):

        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        x = F.softmax(x, dim=-1)
        return x


# %%
def train(
    model: nn.Module,
    x: Tensor,
    y: Tensor,
    x_test: Tensor,
    y_test: Tensor,
    epochs: int = 100,
) -> np.ndarray:

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.1,
        betas=(0.9, 0.99),
        eps=1e-10,
    )
    loss_func = nn.CrossEntropyLoss()
    # loss_func = nn.NLLLoss()
    test_list = np.zeros(epochs)  # Store loss history

    model.train()  # Set model to training mode
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)  # Initialise gradient
        output = model(x)  # Forward pass
        loss = loss_func(output, y)  # Calculate loss
        loss.backward()  # Backward pass
        optimizer.step()  # Optimise weights

        y_pred = model(x).argmax(dim=1)
        acc = (y_pred == y).sum().item() / len(y)
        test_list[epoch] = (
            model(x_test).argmax(dim=1) == y_test
        ).sum().item() / len(y_test)

        print(
            f"Epoch: {epoch + 1}/{epochs}, "
            f"Loss: {loss.item():.4f}, "
            f"Train accuracy: {acc:.2f}, "
            f"Test accuracy: {test_list[epoch]:.2f}"
        )

    return test_list


# %%
k = 3
fold_len = len(x) // k
epochs = 100
test_list = np.zeros((k, epochs))

for i in range(k):
    test_idx = list(range(i * fold_len, (i + 1) * fold_len))
    train_idx = list(range(0, i * fold_len)) + list(
        range((i + 1) * fold_len, k * fold_len)
    )

    test_list[i] = train(
        QNN(),
        x[train_idx],
        y[train_idx],
        x[test_idx],
        y[test_idx],
        epochs=epochs,
    )


# %%
