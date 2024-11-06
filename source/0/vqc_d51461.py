# https://github.com/shaunlee8/quantum-ml/blob/116fdf495d9adc49accaf50c39d81a33076d560b/vqc.py
"""
This code is a part of Qiskit

© Copyright IBM 2017, 2022.

This code is licensed under the Apache License, Version 2.0. You may obtain a copy of this license in
the LICENSE.txt file in the root directory of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.

Any modifications or derivative works of this code must retain this copyright notice, and modified files
need to carry a notice indicating that they have been altered from the originals.
"""
from abc import ABC

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tensorflow as tf
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
import pennylane as qml
from pennylane.templates.state_preparations import MottonenStatePreparation
from pennylane.templates.layers import StronglyEntanglingLayers
from pennylane import numpy as p_np
from qiskit import *
import numpy as np
from qiskit.algorithms.optimizers import COBYLA
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit.circuit.library import PauliFeatureMap, EfficientSU2
from qiskit.circuit import QuantumCircuit, ParameterVector
from keras.datasets import mnist
import time
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()


def ridge_regression(x_train, y_train, x_test, y_test, _lambda=1e-4):
    def normalize(train_x, test_x):
        train_x = train_x.reshape(-1, 784)
        test_x = test_x.reshape(-1, 784)
        train_x = train_x / 255
        test_x = test_x / 255
        return train_x, test_x

    def train(x, y, _lambda):
        n, d = x.shape
        reg_matrix = _lambda * np.eye(d)
        return np.linalg.solve(x.T @ x + reg_matrix, x.T @ y)

    def predict(x, w):
        Y = x.dot(w)
        return np.argmax(Y, axis=1)

    def one_hot(y, num_classes):
        n, k = len(y), num_classes
        Y, i = np.zeros((n, k)), 0
        for num in y:
            Y[i, num] = 1
            i += 1
        return Y

    # Train the ridge regressor and predict targets
    x_train, x_test = normalize(x_train, x_test)
    y_train_one_hot = one_hot(y_train.reshape(-1), 10)
    w_hat = train(x_train, y_train_one_hot, _lambda)
    y_train_pred = predict(x_train, w_hat)
    y_test_pred = predict(x_test, w_hat)

    # Report the train and test errors
    print('Ridge Regression:')
    print(f"\tTrain Error: {np.average(1 - np.equal(y_train_pred, y_train)) * 100:.6g}%")
    print(f"\tTest Error: {np.average(1 - np.equal(y_test_pred, y_test)) * 100:.6g}%")


def vqc_binary(x_train, y_train, x_test, y_test, seed=43):

    # Set parameters
    algorithm_globals.random_seed = seed
    num_qubits = 2
    train_size = 1000
    test_size = 400
    svd_tsne_size = 5000

    def vectorize(train_x, test_x):
        train_x = train_x.reshape(-1, 784)
        test_x = test_x.reshape(-1, 784)
        return train_x, test_x

    def one_hot(y, num_classes):
        n, k = len(y), num_classes
        Y, i = np.zeros((n, k)), 0
        for num in y:
            Y[i, num] = 1
            i += 1
        return Y

    def extractor(train_x, train_y, svd_tsne_size):
        extracted, targets = [], []
        for i in range(svd_tsne_size):
            if train_y[i] == 0:
                extracted.append(train_x[i])
                targets.append(0)
            if train_y[i] == 1:
                extracted.append(train_x[i])
                targets.append(1)
        extracted = np.array(extracted)
        targets = np.array(targets)
        return extracted, targets

    def callback_graph(weights, obj_func_eval):
        objective_func_vals.append(obj_func_eval)
        print("Step: " + str(len(objective_func_vals)))

    # Perform SVD and SNE dimensionality reduction
    x_train, x_test = vectorize(x_train[:svd_tsne_size, :], x_test[:svd_tsne_size, :])
    truncate_svd = TruncatedSVD(n_components=10)
    x_svd_train = truncate_svd.fit_transform(x_train)
    x_svd_test = truncate_svd.fit_transform(x_test)
    print('svd train shape: ' + str(x_svd_train.shape))
    print('svd train shape: ' + str(x_svd_test.shape))
    t_sne = TSNE(n_components=2)
    x_reduced_train = t_sne.fit_transform(x_svd_train)
    print('tsne train shape: ' + str(x_reduced_train.shape))
    x_reduced_test = t_sne.fit_transform(x_svd_test)
    print('tsne test shape: ' + str(x_reduced_test.shape))

    # Extract 0s and 1s from MNIST and normalize
    extracted, targets = extractor(x_reduced_train, y_train, svd_tsne_size)
    print('tar: ' + str(targets))
    print('tar2: ' + str(targets.shape))
    extracted_test, targets_test = extractor(x_reduced_test, y_test, svd_tsne_size)
    extracted = MinMaxScaler().fit_transform(extracted)
    extracted_test = MinMaxScaler().fit_transform(extracted_test)

    # Normalize train and test data
    x_train = MinMaxScaler().fit_transform(x_reduced_train)
    x_test = MinMaxScaler().fit_transform(x_reduced_test)

    # Initialize quantum instance
    qi = QuantumInstance(
        Aer.get_backend("aer_simulator"),
        shots=1024,
        seed_simulator=algorithm_globals.random_seed,
        seed_transpiler=algorithm_globals.random_seed,
    )

    # Initialize optimizer, ansatz, feature_map
    cobyla = COBYLA(maxiter=50)
    feature_map, var_circuit = custom_vqc(num_qubits)

    # Initialize VQC object
    # vqc = VQC(num_qubits=2, optimizer=cobyla, quantum_instance=qi, callback=callback_graph)
    vqc = VQC(num_qubits=num_qubits, ansatz=var_circuit, feature_map=feature_map,
              optimizer=cobyla, quantum_instance=qi, callback=callback_graph)

    print('x_trainshape: ' + str(x_train[:train_size, :].shape))
    print('y_trainshape: ' + str(y_train[:train_size].shape))
    print('extracted_trainshape: ' + str(extracted[:train_size, :].shape))
    print('targets_trainshape: ' + str(targets[:train_size].shape))
    print('extracted_testshape: ' + str(extracted_test[:test_size, :].shape))
    print('targets_testshape: ' + str(targets_test[:test_size].shape))

    # One-hot encoding on targets
    y_train_one_hot = one_hot(y_train.reshape(-1), 10)
    targets_one_hot = tf.keras.utils.to_categorical(targets, num_classes=num_qubits)
    targets_test_one_hot = tf.keras.utils.to_categorical(targets_test, num_classes=num_qubits)
    print('y_hotshape: ' + str(y_train_one_hot.shape))
    print('targets_hotshape: ' + str(targets_one_hot.shape))
    print('targets_test_hotshape: ' + str(targets_test_one_hot.shape))

    # Initialize callback storage
    objective_func_vals = []

    # Fit vqc to data
    # vqc.fit(x_train[:train_size, :], y_train_one_hot[:train_size])
    vqc.fit(extracted[:train_size, :], targets_one_hot[:train_size])

    # Predict train data from input
    vqc.score(extracted[:train_size, :], targets_one_hot[:train_size])
    predict = vqc.predict(extracted[:train_size, :])
    predicted_nums = np.argmax(predict, axis=1)
    print('train predict shape: ' + str(predict.shape))
    print('train predicted nums: ' + str(predicted_nums))
    print('train actual nums: ' + str(targets[:train_size]))

    # Predict test data from input
    predict_test = vqc.predict(extracted_test[:test_size, :])
    predicted_nums_test = np.argmax(predict_test, axis=1)
    print('test predict shape: ' + str(predict_test.shape))
    print('test predicted nums: ' + str(predicted_nums_test))
    print('test actual nums: ' + str(targets_test[:test_size]))

    # Plot the minimization of the optimizer
    plt.title("Objective Function Value against Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Objective Function Value")
    plt.plot(range(len(objective_func_vals)), objective_func_vals)
    plt.show()

    # Report the train and test errors
    print('VQC:')
    print(f"\tTrain Error: {np.average(1 - np.equal(predicted_nums, targets[:train_size])) * 100:.6g}%")
    print(f"\tTest Error: {np.average(1 - np.equal(predicted_nums_test, targets_test[:test_size])) * 100:.6g}%")

    # Save weights to separate file
    vqc.save("vqc_weights_2.pt")


class Autoencoder(tf.keras.Model):

    def __init__(self, dimensions):
        super(Autoencoder, self).__init__()
        self.dimensions = dimensions
        self.encode = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(196, activation='elu'),
            tf.keras.layers.Dense(64, activation='elu'),
            tf.keras.layers.Dense(dimensions, activation='sigmoid'),
        ])
        self.decode = tf.keras.Sequential([
            tf.keras.layers.Dense(dimensions, activation='elu'),
            tf.keras.layers.Dense(196, activation='elu'),
            tf.keras.layers.Dense(784, activation='elu'),
            tf.keras.layers.Reshape((28, 28))
        ])

    def call(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded


def init_autoencoder(num_qubits, x_train, x_test):
    dimensions = num_qubits
    autoencoder = Autoencoder(dimensions)
    autoencoder.compile(optimizer='rmsprop', loss='huber_loss', metrics=["accuracy"])
    autoencoder.fit(x_train, x_train, epochs=20, shuffle=True, validation_data=(x_test, x_test))
    return autoencoder


def vqc_multiclass(x_train, y_train, x_test, y_test):

    def extractor(train_x, train_y):
        extracted, targets = [], []
        for i in range(len(train_y)):
            if train_y[i] == 0 or train_y[i] == 1 or \
                    train_y[i] == 2 or train_y[i] == 3:
                extracted.append(train_x[i])
                targets.append(train_y[i])
        extracted = np.array(extracted)
        targets = np.array(targets)
        return extracted, targets

    def callback_graph(weights, obj_func_eval):
        objective_func_vals.append(obj_func_eval)
        print("Step: " + str(len(objective_func_vals)))

    # Define parameters
    num_qubits = 6
    train_size = 1000
    test_size = 500

    # Extract 4 classes (0, 1, 2, 3)
    x_train, y_train = extractor(x_train, y_train)
    x_test, y_test = extractor(x_test, y_test)

    # Initialize autoencoder
    autoencoder = init_autoencoder(64, x_train, x_test)

    # Reduce MNIST dimensions 784 -> 64
    ex_train = autoencoder.encode(x_train).numpy()
    ex_test = autoencoder.encode(x_train).numpy()

    # Reduce MNIST dimensions 64 -> num_qubits
    truncate_svd = TruncatedSVD(n_components=num_qubits)
    ex_train = truncate_svd.fit_transform(ex_train)
    ex_test = truncate_svd.fit_transform(ex_test)
    print('svd train shape: ' + str(ex_train.shape))
    print('svd test shape: ' + str(ex_test.shape))

    # Normalize the data
    ex_train = MinMaxScaler().fit_transform(ex_train)
    ex_test = MinMaxScaler().fit_transform(ex_test)

    # Perform one-hot encoding with keras
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_qubits)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=num_qubits)

    # Initialize quantum instance
    qi = QuantumInstance(
        Aer.get_backend("aer_simulator"),
        shots=1024,
        seed_simulator=algorithm_globals.random_seed,
        seed_transpiler=algorithm_globals.random_seed,
    )

    # Initialize optimizer, ansatz, feature_map
    cobyla = COBYLA(maxiter=50)
    qc, ansatz = custom_vqc(num_qubits)

    # Initialize VQC object
    vqc = VQC(num_qubits=num_qubits, ansatz=ansatz, feature_map=qc,
              optimizer=cobyla, quantum_instance=qi, callback=callback_graph)

    # Initialize callback storage
    objective_func_vals = []

    # device = qml.device('default.qubit', wires=num_qubits)

    # @qml.qnode(device)
    # def qc_layer(inputs, x):
    #     qml.from_qiskit(qc)({x: inputs})
    #     return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

    # weight_shapes = {"weights": (num_layers, num_qubits, 3)}
    # input_layer = tf.keras.layers.Input(shape=(2 ** num_qubits,))
    # vqc_layer = qml.qnn.KerasLayer(qc_layer, weight_shapes, output_dim=num_qubits)(input_layer)
    # output_layer = tf.keras.layers.Dense(num_classes, activation='softmax')(vqc_layer)

    # tf_vqc = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    # tf_vqc.compile(
    #     loss='categorical_crossentropy',
    #     optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    #     metrics=[tf.keras.metrics.CategoricalAccuracy()], run_eagerly=True
    # )

    # Fit vqc to data
    print('ex_trainshape: ' + str(ex_train.shape))
    print('y_trainshape: ' + str(y_train.shape))
    vqc.fit(ex_train[:train_size, :], y_train[:train_size, :])

    # Predict train data from input
    score = vqc.score(ex_train[:train_size, :], y_train[:train_size])
    print('model acc: ' + str(score))
    predict = vqc.predict(ex_train[:train_size, :])
    predicted_nums = np.argmax(predict, axis=1)
    actual_nums = np.argmax(y_train, axis=1)
    print('train predict shape: ' + str(predict.shape))
    print('train predicted nums: ' + str(predicted_nums))
    print('train actual nums: ' + str(actual_nums[:train_size]))

    # Predict test data from input
    predict_test = vqc.predict(ex_test[:test_size, :])
    predicted_nums_test = np.argmax(predict_test, axis=1)
    actual_nums_test = np.argmax(y_train, axis=1)
    print('test predict shape: ' + str(predict_test.shape))
    print('test predicted nums: ' + str(predicted_nums_test))
    print('test actual nums: ' + str(actual_nums_test[:train_size]))

    # Report the train and test errors
    print('Multi VQC:')
    print(f"\tTrain Error: {np.average(1 - np.equal(predicted_nums, actual_nums[:train_size])) * 100:.6g}%")
    print(f"\tTest Error: {np.average(1 - np.equal(predicted_nums_test, actual_nums_test[:test_size])) * 100:.6g}%")

    # Plot the minimization of the optimizer
    plt.title("Objective Function Value against Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Objective Function Value")
    plt.plot(range(len(objective_func_vals)), objective_func_vals)
    plt.show()


def draw_custom_vqcs():
    # Method for experimenting with VQCs

    num_qubits = 3
    repeat = 1
    x = ParameterVector('x', length=num_qubits)
    qc = QuantumCircuit(num_qubits)

    for r in range(repeat):
        for i in range(num_qubits):
            qc.rx(x[i], i)
        for i in range(num_qubits):
            for j in range(i + 1, num_qubits):
                qc.cx(i, j)
                qc.p(x[i] * x[j], j)
                qc.cx(i, j)
    print(qc)

    feature_dim = 3
    pauli_feature_map = PauliFeatureMap(feature_dimension=feature_dim, reps=3, paulis=['Z', 'X', 'ZY'])
    print(pauli_feature_map)

    return qc


def custom_vqc(num_qubits, repeats=3):
    x = ParameterVector('x', length=num_qubits)
    qc = QuantumCircuit(num_qubits)

    var_circuit = EfficientSU2(num_qubits, entanglement='linear', reps=2, insert_barriers=True)
    var_circuit.draw(output='mpl')
    plt.show()

    # # Test 1
    # for r in range(repeats):
    #     for i in range(num_qubits):
    #         qc.rz(x[i], i)
    #         qc.rx(x[i], i)
    #     for control in range(num_qubits - 1, 0, -1):
    #         target = control - 1
    #         qc.rx(x[target], target)
    #         qc.cx(control, target)
    #         qc.rx(x[target], target)
    #     for i in range(num_qubits):
    #         qc.rz(x[i], i)
    #         qc.rx(x[i], i)

    # Test 2 - Good for binary
    for r in range(repeats):
        for i in range(num_qubits):
            qc.rx(x[i], i)
            qc.rz(x[i], i)
        for i in range(num_qubits - 1, 0, -1):
            qc.cx(i, i - 1)

    qc.draw(output='mpl')
    plt.show()
    print(x.params)

    return qc, var_circuit


if __name__ == '__main__':

    # Load MNIST dataset
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    print('X_train: ' + str(train_X.shape))
    print('Y_train: ' + str(train_y.shape))
    print('X_test:  ' + str(test_X.shape))
    print('Y_test:  ' + str(test_y.shape))

    # Perform ridge regression
    # ridge_regression(train_X, train_y, test_X, test_y)

    # Perform binary classification
    # vqc_binary(train_X, train_y, test_X, test_y)

    # Perform multiclass classification
    vqc_multiclass(train_X, train_y, test_X, test_y)

    # Draw circuits
    # custom_vqc()
