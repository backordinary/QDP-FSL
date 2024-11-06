# https://github.com/boyesjo/tma4500/blob/41c56c2cd29e3763832334701fd2c676f2cb4bc8/code/qcnn/test.py
# %% https://qiskit.org/documentation/machine-learning/tutorials/11_quantum_convolutional_neural_networks.html # noqa
import matplotlib.pyplot as plt
import numpy as np
from qiskit import Aer, QuantumCircuit
from qiskit.algorithms import optimizers
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap, ZZFeatureMap
from qiskit.opflow import AerPauliExpectation, PauliSumOp
from qiskit.providers import fake_provider
from qiskit.providers.aer.noise import NoiseModel
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit_machine_learning.algorithms.classifiers import (
    NeuralNetworkClassifier,
)
from qiskit_machine_learning.neural_networks import TwoLayerQNN

algorithm_globals.random_seed = 1337

# %%
# set up noise quantuminstance
noise_model = NoiseModel.from_backend(fake_provider.FakeMontreal())
simulator = Aer.get_backend("aer_simulator")
qi = QuantumInstance(
    simulator,
    shots=1024,
    # noise_model=noise_model,
)


# %%
def generate_data(n: int = 32):
    x = np.zeros((2 * n, 4, 2))
    y = np.zeros(2 * n, dtype=int)

    for i in range(n):
        x[i, i % 4] = 1
        y[i] = 1

    for i in range(n):
        x[n + i][:, i % 2] = 1
        y[n + i] = -1

    # flatten images
    x = x.reshape(2 * n, 8)
    # rescale to (0, pi/2)
    x *= np.pi / 2

    # add gaussian noise
    x += algorithm_globals.random.normal(0, 0.1, x.shape)

    # shuffle data
    idx = np.arange(2 * n)
    algorithm_globals.random.shuffle(idx)
    x = x[idx]
    y = y[idx]

    return x, y


x_train, y_train = generate_data(32)
x_test, y_test = generate_data(8)


# %%
# plot first 8 images
fig, axes = plt.subplots(2, 4, figsize=(6, 2))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(x_train[i].reshape(4, 2).T, cmap="gray")
    # ax.set_title(f"Label: {y_train[i]}")
    # remove ticks
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
plt.savefig("data.pdf")


# %%
def conv_circuit(params):
    target = QuantumCircuit(2)
    target.rz(-np.pi / 2, 1)
    target.cx(1, 0)
    target.rz(params[0], 0)
    target.ry(params[1], 1)
    target.cx(0, 1)
    target.ry(params[2], 1)
    target.cx(1, 0)
    target.rz(np.pi / 2, 0)
    return target


def conv_layer(num_qubits, param_prefix):
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        qc = qc.compose(
            conv_circuit(params[param_index : (param_index + 3)]), [q1, q2]
        )
        qc.barrier()
        param_index += 3
    for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
        qc = qc.compose(
            conv_circuit(params[param_index : (param_index + 3)]), [q1, q2]
        )
        qc.barrier()
        param_index += 3

    qc_inst = qc.to_instruction()

    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, qubits)
    return qc


def pool_circuit(params):
    target = QuantumCircuit(2)
    target.rz(-np.pi / 2, 1)
    target.cx(1, 0)
    target.rz(params[0], 0)
    target.ry(params[1], 1)
    target.cx(0, 1)
    target.ry(params[2], 1)

    return target


def pool_layer(sources, sinks, param_prefix):
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for source, sink in zip(sources, sinks):
        qc = qc.compose(
            pool_circuit(params[param_index : (param_index + 3)]),
            [source, sink],
        )
        qc.barrier()
        param_index += 3

    qc_inst = qc.to_instruction()

    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, range(num_qubits))
    return qc


# %%
feature_map = ZFeatureMap(feature_dimension=8)
feature_map.decompose().draw(output="mpl")

# %%
ansatz = QuantumCircuit(8)
ansatz.compose(conv_layer(8, "conv1"), range(8), inplace=True)
ansatz.compose(
    pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "pool1"), range(8), inplace=True
)
ansatz.compose(conv_layer(4, "conv2"), range(4, 8), inplace=True)
ansatz.compose(pool_layer([0, 1], [2, 3], "pool2"), range(4, 8), inplace=True)
ansatz.compose(conv_layer(2, "conv3"), range(6, 8), inplace=True)
ansatz.compose(pool_layer([0], [1], "pool3"), range(6, 8), inplace=True)
observable = PauliSumOp.from_list([("Z" + "I" * 7, 1)])
circuit = QuantumCircuit(8)
circuit.compose(feature_map, range(8), inplace=True)
circuit.compose(ansatz, range(8), inplace=True)
qnn = TwoLayerQNN(
    8,
    feature_map=feature_map,
    ansatz=ansatz,
    observable=observable,
    exp_val=AerPauliExpectation(),
    quantum_instance=qi,
)

circuit.draw("mpl")


# %%
def score(weights, x, y):
    y_pred = qnn.forward(x, weights=weights)
    y_pred = np.where(y_pred > 0, 1, -1).flatten()
    return np.mean(y_pred == y)


def callback(weights, loss):
    stats = {
        "iteration": len(history),
        "loss": loss,
        "training_acc": score(weights, x_train, y_train),
        "test_acc": score(weights, x_test, y_test),
    }
    # print stats with 2 decimal places
    print(
        f"iteration: {stats['iteration']:4}, "
        f"loss: {stats['loss']:4.3f}, "
        f"training_acc: {stats['training_acc']:3.2f}, "
        f"test_acc: {stats['test_acc']:3.2f}"
    )
    history.append(stats)


init_weights = algorithm_globals.random.random(qnn.num_weights)

opflow_classifier = NeuralNetworkClassifier(
    qnn,
    loss="absolute_error",
    optimizer=optimizers.COBYLA(maxiter=400),
    # optimizer=optimizers.ADAM(lr=0.01, maxiter=100, snapshot_dir="garbage"),
    # optimizer=optimizers.GradientDescent(learning_rate=0.1),
    callback=callback,
    initial_point=init_weights,
)

history = []
opflow_classifier.fit(x_train, y_train)
# %%
y_predict = opflow_classifier.predict(x_test)
x = np.asarray(x_test)
y = np.asarray(y_test)
print(
    "Accuracy from the test data: "
    f"{np.round(100 * opflow_classifier.score(x, y), 2)}%"
)

# plot 6 test images
fig, axes = plt.subplots(2, 3, figsize=(10, 10))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(x_test[i].reshape(4, 2), cmap="gray")
    ax.set_title(
        "predicted horizontal" if y_predict[i] == 1 else "predicted vertical"
    )
    ax.axis("off")

# # %%
# import pandas as pd

# noisy_optim = pd.read_csv("noisy_optim.csv", names=["Noisy"])
# optim = pd.Series(objective_func_vals, name="Pure")
# df = pd.concat([noisy_optim, optim], names=["Noisy", "Pure"], axis=1)
# df.plot()

# # save df to csv
# df.to_csv("optim.csv", index=True)
# # %%

# %%
