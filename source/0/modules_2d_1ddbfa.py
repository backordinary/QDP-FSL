# https://github.com/koritsky/ferrimagnet/blob/4170fea711d2ef960b39e6be02de567e8d3dac79/confusion_learning/modules_2D.py
import qiskit.quantum_info
import numpy as np 
import torch
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import pyplot
import matplotlib.pyplot as plt 
from xgboost import XGBClassifier

def step_gen(X, y, transition=0.,k = 1, slope=100):
    X = X.reshape(-1, 1)
    noise = 0.001 * np.random.randn(*X.shape)

    return 1 / (1 + np.exp(-slope * (X - k * y - transition + noise))) + noise


def data_labeling(data, params, p_expect):
    labels = (params > p_expect).astype('float')

    data_mean = np.mean(data, axis=0)
    data_std = np.std(data, axis=0)
    data = (data - data_mean) / (data_std + 0.01)

    return data, labels


def XGB_learning(data, labels):
    data_train, data_test, labels_train, labels_test = \
        train_test_split(data, labels, test_size=0.2, random_state=7)

    # fit model no training data

    model = XGBClassifier()
    eval_set = [(data_train, labels_train), (data_test, labels_test)]
    model.fit(data_train, labels_train, eval_metric=["error", "logloss"], eval_set=eval_set, verbose=False)

    labels_pred = model.predict(data_test)
    predictions = [round(value) for value in labels_pred]

    results = model.evals_result()
    accuracy = accuracy_score(labels_test, predictions)
    learn_curve = 1 - np.array(results['validation_1']['error'])
    # print('Accuracy = ', accuracy)

    return accuracy, learn_curve

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
        acc, curve = XGB_learning(data, labels)
        w_data.append(acc)
        learn_curves.append(curve)
    return w_data, learn_curves


def mainloop(bound, p_true, samples, X_dotn, Y_dotn, k=-1):

    print('w-shape in bounds = ', bound)
    X_params = np.linspace(bound[0], bound[1], X_dotn)
    Y_params = np.linspace(bound[0], bound[1], Y_dotn)

    Z = np.zeros((X_dotn, Y_dotn))
    for i, y in tqdm(enumerate(Y_params)):
        w_data_stack = []
        for _ in tqdm(range(samples)):
            # print('------- w-shape sample number =', i, '-------')
            raw_data = step_gen(X_params, y, transition=p_true, k=k)
            w_data, learn_curves = w_shape_gen(raw_data, X_params)

            w_data_stack.append(w_data)
            learn_curves_data.append(learn_curves)

        w_shape = np.mean(w_data_stack, axis=0)

        Z[:, i] = w_shape #sry, I think this could be better

        # w_bar = np.std(w_data_stack, axis=0)
        # learn_curves = np.mean(learn_curves_data, axis=0)



    return X_params, Y_params, Z



if __name__ == '__main__':
    bounds = [-1, 1]
    samples = 20
    dots_number = 100
    p_true = 0

    out = mainloop(bounds, p_true, samples, dots_number)

    X_theory = [-1, -1 / 2, 0, 1 / 2, 1]
    Y_theory = [1, 0.75, 1, 0.75, 1]
    plot_wshape(out, (X_theory, Y_theory), bar_flag=True)
