# https://github.com/koritsky/ferrimagnet/blob/4170fea711d2ef960b39e6be02de567e8d3dac79/confusion_learning/modules_2D_ferro.py
import qiskit.quantum_info
import numpy as np 
import torch
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from matplotlib import pyplot
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

from confusion_learning.energy import energy
from confusion_learning.w_find import *

def step_gen(X, y, transition=0.,k = 1, slope=100):
    X = X.reshape(-1, 1)
    noise = 0.001 * np.random.randn(*X.shape)

    return 1 / (1 + np.exp(-slope * (X - k * y - transition + noise))) + noise

def energy_gen(x, h, n_thetas=100, energy_func=energy):
    # X is a list of impurity concentration
    # h is a parameter of magnetic field

    # here we generate range of energies with given X and h
    # (you can change other parameters of task in confusion_learning/energy.py
    Thetas = np.sort(np.random.rand(n_thetas) * np.pi)
    # Thetas = np.linspace(0, 2 * np.pi, n_thetas)
    Energies = energy_func(Thetas, x, h)
    e_max = np.max(Energies)
    e_min = np.min(Energies)

    return np.argmin((Energies - e_min) / (e_max - e_min))

def data_labeling(data, params, p_expect):
    labels = (params > p_expect).astype('float')
    # data_mean = np.mean(data, axis=0)
    # data_std = np.std(data, axis=0)
    # data = (data - data_mean) / (data_std + 0.01)

    return data, labels


def XGB_learning(data, labels):
    data_train, data_test, labels_train, labels_test = \
        train_test_split(data, labels, test_size=0.3)

    # Shitty bug fix when labels_train consists class that is not in labels_test
    if len(np.unique(labels_train)) == 1:
        labels_test = labels_train[0] * np.ones(labels_test.shape)

    # fit model no training data
    model = XGBClassifier()
    eval_set = [(data_test, labels_test)]

    model.fit(data_train, labels_train, eval_metric=["error", "logloss"], eval_set=eval_set, verbose=False)

    labels_pred = [round(value) for value in model.predict(data_test)]

    accuracy = accuracy_score(labels_test, labels_pred)
    # print('Accuracy = ', accuracy)

    return accuracy

def plot_learn_curves_cut(learn_curves, p_guess, x_true):
    fig = plt.figure()
    plt.rc('font', family='serif')
    cmap = plt.get_cmap('twilight')
    ax = fig.add_subplot(1, 1, 1)
    fig.set_size_inches(6, 3)
    leng = len(learn_curves[0, :])
    for i in range(leng):
        ax.plot(p_guess, learn_curves[:, i], color=cmap(float(i)/leng))
    ax.axvline(x=x_true)
    ax.set_xlabel('p', fontsize=15)
    ax.set_ylabel('Accuracy', fontsize=15)
    ax.grid()
    plt.show()


def w_shape_gen(data, params):
    w_data, learn_curves = [], []

    for p in params:
        data, labels = data_labeling(data, params, p)
        accuracy = XGB_learning(data, labels)
        w_data.append(accuracy)
    return w_data


def mainloop(X, Y, n_thetas=100, n_samples=10, energy_func=energy):

    Z = np.zeros((X.shape[0], Y.shape[0])) # calculated accuracy
    Z_nearest = np.zeros((X.shape[0], Y.shape[0])) # closest w_shape
    C = np.zeros(Y.shape[0])
    for i, y in tqdm(enumerate(Y)):
        w_data_stack = []
        for sample in tqdm(range(n_samples)):
            # print('------- w-shape sample number =', i, '-------')

            raw_data = np.array([energy_gen(x, y, n_thetas, energy_func=energy_func) for x in X]).reshape(-1, 1)
            # raw_data = np.array([energy_gen(x, h, n_thetas, energy_func=energy_func) for x in X]).reshape(len(X), n_thetas)
            w_data = w_shape_gen(raw_data, X)

            w_data_stack.append(w_data)
            # learn_curves_data.append(learn_curves)
        w_shape = np.mean(w_data_stack, axis=0)
        nearest_wshape, best_c = w_find(X[0], X[-1], w_shape)
        Z[:, i] = w_shape #sry, I think this could be better
        Z_nearest[:, i], C[i] = nearest_wshape, best_c
        # w_bar = np.std(w_data_stack, axis=0)
        # learn_curves = np.mean(learn_curves_data, axis=0)

    return Z, Z_nearest, C



if __name__ == '__main__':
    X = np.linspace(0, 1, 50)

    h = 10 ** 3
    Energies = energy_gen(X, h)
    print(Energies)



