# https://github.com/ankit27kh/Explorations_In_QML/blob/9c999c9e972d87608292fea08a1a055e81fe14a9/rotor_noise_QISKIT_after_data.py
import itertools
import time
import pennylane as qml
import numpy as np
import qiskit.algorithms.optimizers
from qiskit.providers.aer import AerSimulator
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from scipy.linalg import expm
from qiskit import QuantumCircuit, transpile, execute
from qiskit.test.mock import FakeLagos

seed = 4
np.random.seed(seed)
qiskit.utils.algorithm_globals.random_seed = seed

p = np.pi / 2
steps = 100
num_points = 5
k = 1
num_layers = 1
epochs = 10
lr = 0.1
use_noise = False
shots = 1000
J = 1

device_backend = FakeLagos()

if use_noise:
    print("NOISE ON")
    backend = AerSimulator.from_backend(device_backend)
else:
    print("NOISE OFF")
    backend = AerSimulator()

pauli_x = np.array([[0, 1], [1, 0]])
pauli_y = np.array([[0, -1j], [1j, 0]])
pauli_z = np.array([[1, 0], [0, -1]])

num_qubits = int(2 * J)


def get_Ji(n, pauli_i):
    matrix = np.zeros([2 ** n, 2 ** n])
    for i in range(1, n + 1):
        matrix = matrix + np.kron(
            np.identity(2 ** (i - 1)), np.kron(pauli_i, np.identity(2 ** (n - i)))
        )
    return matrix / 2


def get_J_z_2(n):
    matrix = np.zeros([2 ** n, 2 ** n])
    for j in range(2, n + 1):
        for i in range(1, j):
            left = np.identity(2 ** (i - 1))
            middle = np.identity(2 ** (j - 1 - i))
            right = np.identity(2 ** (n - j))
            matrix = matrix + np.kron(
                left, np.kron(pauli_z, np.kron(middle, np.kron(pauli_z, right)))
            )
    return matrix / 2


J_x = get_Ji(num_qubits, pauli_x)
J_y = get_Ji(num_qubits, pauli_y)
J_z = get_Ji(num_qubits, pauli_z)
J_z_2 = get_J_z_2(num_qubits)

floquet = expm(-1j * k / 2 / J * J_z_2) @ expm(-1j * p * J_y)

init_theta = np.linspace(0, np.pi, num_points)
init_phi = np.linspace(-np.pi, np.pi, num_points)

X = np.array(list(itertools.product(init_theta, init_phi)))

# Z-Hemispheres
y = np.array([-1 if i[0] <= np.pi / 2 else 1 for i in X])

# X-Hemispheres
# y = np.array([-1 if abs(i[1]) <= np.pi / 2 else 1 for i in X])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

state_creation_dev = qml.device("default.qubit", wires=num_qubits, shots=None)


@qml.qnode(state_creation_dev)
def state_creation(x):

    initial_angles = np.array([x[0], x[1], 0])

    qml.broadcast(
        qml.U3,
        wires=range(num_qubits),
        pattern="single",
        parameters=np.array([initial_angles] * num_qubits),
    )

    for _ in range(steps):
        qml.QubitUnitary(floquet, wires=range(num_qubits))

    return qml.state()


initial_states_train = []
initial_states_test = []

t1 = time.time()
for x in X_train:
    initial_states_train.append(state_creation(x).tolist())
for x in X_test:
    initial_states_test.append(state_creation(x).tolist())
print("States Created", time.time() - t1)


def classifier(state, params):
    params = params.reshape([num_layers, num_qubits, 3])

    qc = QuantumCircuit(num_qubits, 1)
    qc.initialize(state, qc.qubits)

    for l in range(num_layers):
        for i in range(num_qubits):
            qc.u(*params[l][i], i)
        if num_qubits == 2:
            qc.cx(0, 1)
        else:
            for i in range(num_qubits - 1):
                qc.cx(i, i + 1)
            qc.cx(num_qubits - 1, 0)

    qc.measure(0, 0)

    return qc


weights = np.random.uniform(
    -np.pi,
    np.pi,
    [num_layers, num_qubits, 3],
)


def circuit(state, params):
    qc = classifier(state, params)
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
    ep = (res.get("1", 0) - res.get("0", 0)) / shots
    return ep


def cost(parameters, x, y):
    p = circuit(x, parameters)
    return (y - p) ** 2


def predict_one(x, parameters):
    ep = circuit(x, parameters)
    return 2 * (ep >= 0) - 1


def all_cost(parameters, x, y):
    error = 0
    for xi, yi in zip(x, y):
        error = error + cost(parameters, xi, yi)
    return error / len(x)


def all_predict(x, parameters):
    y = []
    for xi in x:
        y.append(predict_one(xi, parameters))
    return np.array(y)


opt = qiskit.algorithms.optimizers.SPSA(
    maxiter=epochs,
    learning_rate=lr,
    perturbation=lr * 2,
    # second_order=True,
    # blocking=True,
)
t1 = time.time()

res = opt.optimize(
    weights.ravel().shape[0],
    lambda params: all_cost(params, initial_states_train, y_train),
    initial_point=weights.ravel(),
)

print("Optimization done", time.time() - t1)

trained_weights = res[0].reshape([num_layers, num_qubits, 3])

print("Scores after training:")
y_predict_train = all_predict(initial_states_train, trained_weights)
y_predict_test = all_predict(initial_states_test, trained_weights)
acc_train = accuracy_score(y_pred=y_predict_train, y_true=y_train)
acc_test = accuracy_score(y_pred=y_predict_test, y_true=y_test)
print(
    "Training Data:",
    acc_train,
    "Testing Data:",
    acc_test,
)
"""
import pickle

save_data = [use_noise, num_points, acc_train, acc_test]
with open(f'new_data_/rotor/noise/after_data_{use_noise}_{num_points}.pkl', 'wb') as file:
    pickle.dump(save_data, file)
print(save_data)
"""
