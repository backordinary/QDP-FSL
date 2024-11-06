# https://github.com/ankit27kh/Explorations_In_QML/blob/9c999c9e972d87608292fea08a1a055e81fe14a9/kernel_classification_noise_QISKIT.py
import time
import matplotlib.pyplot as plt
import numpy as np
import qiskit.algorithms.optimizers
from qiskit.providers.aer import AerSimulator
from qiskit import QuantumCircuit, transpile, execute
from qiskit.test.mock import FakeLagos
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from datasets import circle
from sklearn.preprocessing import StandardScaler, MinMaxScaler

seed = 42
np.random.seed(seed)
qiskit.utils.algorithm_globals.random_seed = seed

num_points = 50
num_layers = 5
num_qubits = 1
epochs = 50
lr = 0.1
use_noise = False
shots = 1000

device_backend = FakeLagos()

if use_noise:
    print("NOISE ON")
    backend = AerSimulator.from_backend(device_backend)
else:
    print("NOISE OFF")
    backend = AerSimulator()

X_train, X_test, y_train, y_test, dimension = circle(points=num_points, test_size=.8)
X_train, X_test, y_train, y_test = X_train.numpy(), X_test.numpy(), y_train.numpy(), y_test.numpy()

y_train = np.array([-1 if y == 0 else 1 for y in y_train])
y_test = np.array([-1 if y == 0 else 1 for y in y_test])

num_rots = int(np.ceil(dimension / 3))

scaler_x = StandardScaler()
X_train_scaled = scaler_x.fit_transform(X_train)
X_test_scaled = scaler_x.transform(X_test)

scaler_x_mm = MinMaxScaler([-1, 1])
X_train_scaled = scaler_x_mm.fit_transform(X_train_scaled)
X_test_scaled = scaler_x_mm.transform(X_test_scaled)

parameters = np.random.uniform(-np.pi / 2, np.pi / 2, num_layers * num_rots * 3 * 2)

plt.scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], c=y_test, label="og")
plt.legend()
plt.show()


def feature_map(x, params):
    extra = 3 * num_rots - dimension
    x = np.concatenate([x, np.zeros(extra)])
    num_thetas = num_layers * num_rots * 3
    varias = params[:num_thetas].reshape([num_layers, num_rots, 3])
    weights = params[num_thetas:].reshape([num_layers, num_rots, 3])

    qc = QuantumCircuit(num_qubits, 1)

    for k in range(num_layers):
        for i in range(num_rots):
            qc.u(*(x[3 * i: 3 * (i + 1)] * weights[k][i] + varias[k][i]), 0)

    return qc


def adjoint_feature_map(x, params):
    extra = 3 * num_rots - dimension
    x = np.concatenate([x, np.zeros(extra)])
    num_thetas = num_layers * num_rots * 3
    varias = params[:num_thetas].reshape([num_layers, num_rots, 3])
    weights = params[num_thetas:].reshape([num_layers, num_rots, 3])

    qc = QuantumCircuit(num_qubits, 1)

    angles_list = []
    for k in range(num_layers):
        for i in range(num_rots):
            angles_list.append((x[3 * i: 3 * (i + 1)] * weights[k][i] + varias[k][i]))

    for i in range(len(angles_list)):
        angles = angles_list[-(i + 1)]
        qc.u(-angles[0], -angles[2], -angles[1], 0)

    return qc


def kernel(x1, x2, params):
    qc_1 = feature_map(x1, params)
    qc_2 = adjoint_feature_map(x2, params)
    qc = qc_1.compose(qc_2)
    qc.measure(0, 0)

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

    k = res.get("0", 0) / shots

    return k


def kernel_matrix(x1, x2, params):
    m = len(x1)
    n = len(x2)
    matrix = np.zeros([m, n])
    for i, p1 in enumerate(x1):
        for j, p2 in enumerate(x2):
            matrix[i][j] = kernel(p1, p2, params)

    return matrix


training_kernel_matrix = kernel_matrix(X_train_scaled, X_train_scaled, parameters)
testing_kernel_matrix = kernel_matrix(X_test_scaled, X_train_scaled, parameters)

training_kernel_matrix = (training_kernel_matrix + training_kernel_matrix.T) / 2

svm = SVC(kernel="precomputed")
svm.fit(training_kernel_matrix, y_train.ravel())
y_pred_test = svm.predict(testing_kernel_matrix)
print(accuracy_score(y_test, y_pred_test), 'score before')

plt.scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], c=y_pred_test, label="before")
plt.legend()
plt.show()


def target_alignment(X, Y, params):
    Kx = kernel_matrix(X, X, params)
    _Y = np.array(Y)

    # Rescaling
    class_1 = np.count_nonzero(_Y == 1)
    class_0 = np.count_nonzero(_Y == -1)
    _Y = np.where(_Y == 1, _Y / class_1, _Y / class_0)

    Ky = np.outer(_Y, _Y)

    N = len(X)

    def kernel_center(ker):
        sq = np.ones([N, N])
        c = np.identity(N) - sq / N
        return np.matmul(c, np.matmul(ker, c))

    KxC = kernel_center(Kx)
    KyC = kernel_center(Ky)

    kxky = np.trace(np.matmul(KxC.T, KyC))
    kxkx = np.linalg.norm(KxC)
    kyky = np.linalg.norm(KyC)

    kta = kxky / kxkx / kyky
    return kta


def cost(params, x, y):
    return -target_alignment(x, y, params)


print(-cost(parameters, X_train_scaled, y_train), 'kta before')
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
    lambda params: cost(params, X_train_scaled, y_train),
    initial_point=parameters,
)

print("Optimization done", time.time() - t1)

trained_params = res[0]

training_kernel_matrix = kernel_matrix(X_train_scaled, X_train_scaled, trained_params)
testing_kernel_matrix = kernel_matrix(X_test_scaled, X_train_scaled, trained_params)

training_kernel_matrix = (training_kernel_matrix + training_kernel_matrix.T) / 2

svm = SVC(kernel="precomputed")
svm.fit(training_kernel_matrix, y_train.ravel())
y_pred_test = svm.predict(testing_kernel_matrix)
print(accuracy_score(y_test, y_pred_test), 'score after')

print(-cost(trained_params, X_train_scaled, y_train), 'kta after')

plt.scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], c=y_pred_test, label="after")
plt.legend()
plt.show()
