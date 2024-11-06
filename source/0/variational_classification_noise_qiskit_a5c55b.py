# https://github.com/ankit27kh/Explorations_In_QML/blob/9c999c9e972d87608292fea08a1a055e81fe14a9/variational_classification_noise_QISKIT.py
import time
import numpy as np
import qiskit.algorithms.optimizers
from qiskit.providers.aer import AerSimulator
from qiskit import QuantumCircuit, transpile, execute
from qiskit.test.mock import FakeLagos
from sklearn.metrics import accuracy_score
from datasets import circle
from sklearn.preprocessing import StandardScaler, MinMaxScaler

seed = 42
np.random.seed(seed)
qiskit.utils.algorithm_globals.random_seed = seed

num_points = 100
num_layers = 5
num_qubits = 1
epochs = 10
lr = 0.1
use_noise = True
shots = 1000
eps = 0.0001

device_backend = FakeLagos()

if use_noise:
    print("NOISE ON")
    backend = AerSimulator.from_backend(device_backend)
else:
    print("NOISE OFF")
    backend = AerSimulator()

X_train, X_test, y_train, y_test, dimension = circle(points=num_points, test_size=.8)
X_train, X_test, y_train, y_test = (
    X_train.numpy(),
    X_test.numpy(),
    y_train.numpy(),
    y_test.numpy(),
)

num_rots = int(np.ceil(dimension / 3))

scaler_x = StandardScaler()
X_train_scaled = scaler_x.fit_transform(X_train)
X_test_scaled = scaler_x.transform(X_test)

scaler_x_mm = MinMaxScaler([-1, 1])
X_train_scaled = scaler_x_mm.fit_transform(X_train_scaled)
X_test_scaled = scaler_x_mm.transform(X_test_scaled)


def classifier(x, params):
    extra = 3 * num_rots - dimension
    x = np.concatenate([x, np.zeros(extra)])
    num_thetas = num_layers * num_rots * 3
    varias = params[:num_thetas].reshape([num_layers, num_rots, 3])
    weights = params[num_thetas:].reshape([num_layers, num_rots, 3])

    qc = QuantumCircuit(num_qubits, 1)

    for k in range(num_layers):
        for i in range(num_rots):
            qc.u(*(x[3 * i: 3 * (i + 1)] * weights[k][i] + varias[k][i]), 0)

    qc.measure(0, 0)

    return qc


parameters = np.random.uniform(-np.pi / 2, np.pi / 2, num_layers * num_rots * 3 * 2)


def circuit(x, params):
    qc = classifier(x, params)
    qc = transpile(qc, backend, optimization_level=0, seed_transpiler=seed)
    res = (
        execute(
            qc,
            backend,
            shots=shots,
            seed_simulator=seed,
            seed_transpiler=seed,
            optimization_level=0,
        )
            .result()
            .get_counts()
    )
    p0 = res.get("0", 0) / shots
    p1 = res.get("1", 0) / shots
    return p0, p1


def cost(params, x, y):
    p = circuit(x, params)
    p0 = max(eps, p[0])
    p1 = max(eps, p[1])
    return -np.asarray(y * np.log(p1) + (1 - y) * np.log(p0))


def all_cost(params, x, y):
    error = 0
    for xi, yi in zip(x, y):
        error = error + cost(params, xi, yi)
    return error / len(x)


def all_predict(x, params):
    y = []
    for xi in x:
        p = circuit(xi, params)
        y.append(np.argmax(p))
    return np.asarray(y)


print("Scores before training:")
print(all_cost(parameters, X_train_scaled, y_train), "cost")
y_predict_test = all_predict(X_test_scaled, parameters)
print(accuracy_score(y_test, y_predict_test), "acc")

opt = qiskit.algorithms.optimizers.SPSA(
    maxiter=epochs,
    learning_rate=lr,
    perturbation=lr * 2,
    # second_order=True,
    # blocking=True,
)
t1 = time.time()

res = opt.optimize(
    parameters.shape[0],
    lambda params: all_cost(params, X_train_scaled, y_train),
    initial_point=parameters,
)

print("Optimization done", time.time() - t1)

trained_params = res[0]

print("Scores after training:")
print(all_cost(trained_params, X_train_scaled, y_train), "cost")
y_predict_test = all_predict(X_test_scaled, trained_params)
print(accuracy_score(y_test, y_predict_test), "acc")
