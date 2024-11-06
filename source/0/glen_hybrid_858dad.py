# https://github.com/mrdobson/SENSIP_2021_REU/blob/4fdbfb62497db3060560e06d0988ddd7b2a41308/python/glen_hybrid.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
##############################################################################
# Hybrid quantum-classical QNN implementation
# Glen Uehara
##############################################################################

import numpy as np
import matplotlib.pyplot as plt
import pylatexenc

from qiskit import Aer, QuantumCircuit
from qiskit.opflow import Z, I, StateFn, PauliSumOp, AerPauliExpectation, ListOp, Gradient
from qiskit.utils import QuantumInstance
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.algorithms.optimizers import COBYLA, L_BFGS_B, ADAM

from qiskit_machine_learning.neural_networks import TwoLayerQNN, CircuitQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier, VQC
from qiskit_machine_learning.algorithms.regressors import NeuralNetworkRegressor, VQR

from typing import Union

from qiskit_machine_learning.exceptions import QiskitMachineLearningError
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import confusion_matrix

quantum_instance = QuantumInstance(Aer.get_backend('qasm_simulator'), shots=1024)

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    plt.figure(figsize=(8, 6))
    plt.rcParams.update({'font.size': 15})
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    #plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
        
    print(cm)

    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
#    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()
    #%%
    
#X_dataset=np.loadtxt('fault.txt')
#Y_dataset=np.loadtxt('faultlabel.txt')

#scaler_x = MinMaxScaler()
#scaler_x = StandardScaler()

#X = scaler_x.fit_transform(X_dataset)
num_inputs = 2
num_samples = 20

X = 2*np.random.rand(num_samples, num_inputs) - 1
y01 = 1*(np.sum(X, axis=1) >= 0)  # in { 0,  1}

y = 2*y01-1                 # in {-1, +1}

#Classification with the an OpflowQNN
# construct QNN
# specify the feature map
fm = ZZFeatureMap(num_inputs, reps=2)
fm.draw(output='mpl')

# specify the ansatz
ansatz = RealAmplitudes(num_inputs, reps=4)
ansatz.draw(output='mpl')

# specify the observable
observable = PauliSumOp.from_list([('Z'*num_inputs, 1)])
print(observable)

# define two layer QNN
opflow_qnn = TwoLayerQNN(num_inputs, 
                   feature_map=fm, 
                   ansatz=ansatz, 
                   observable=observable, quantum_instance=quantum_instance)

inputs = np.random.rand(opflow_qnn.num_inputs)
weights = np.random.rand(opflow_qnn.num_weights)

# QNN forward pass
opflow_qnn.forward(inputs, weights)

# QNN backward pass
opflow_qnn.backward(inputs, weights)

# construct neural network classifier
opflow_classifier = NeuralNetworkClassifier(opflow_qnn, optimizer=ADAM(maxiter=50))

# fit classifier to data
opflow_classifier.fit(X, y)

# score classifier
opflow_classifier.score(X, y)

# feed into other parameter, print out -> is accuracy

# evaluate data points
y_predict = opflow_classifier.predict(X)
print(y_predict)
# plot this after done
# if ypred == 0, sum each time it is, print out sum
# find where ypred == 0, find idx, check true y

print('Accuracy:', sum(y_predict == y)/len(y))

cm = confusion_matrix(y_true=y, y_pred=y_predict)

plot_confusion_matrix(cm , 
                      normalize    = False,
                      target_names = ['Normal', 'Faulty'],
                      title        = "Confusion Matrix")
