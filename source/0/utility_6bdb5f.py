# https://github.com/quantumyatra/quantum_computing/blob/5723b8639ffa6130f229d42141d1a108b7dc5fd3/qiskit_textbook/utility.py
#!/usr/bin/env python3

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import Aer, execute, BasicAer

import numpy as np
import matplotlib.pyplot as plt

def get_state_vector(qc):
    backend = BasicAer.get_backend('statevector_simulator')
    res = execute(qc, backend).result()
    return res.get_statevector(qc, decimals=3)
    
def create_bell_pair(qc, a, b):
    qc.h(a)
    qc.cx(a,b)

import numpy as np
import matplotlib.pyplot as plt

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

def breast_cancer_pca(training_size, test_size, n, PLOT_DATA=False):
    class_labels = [r'Benign', r'Malignant']
    
    # First the dataset must be imported.
    cancer = sklearn.datasets.load_breast_cancer()
    
    # Here the data is divided into 70% training, 30% testing.
    X_train, X_test, Y_train, Y_test = train_test_split(cancer.data, cancer.target, test_size=0.3, random_state=109)
    
    # features will be standardized to fit a normal distribution.
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    # To be able to use this data with the given
    # number of qubits, the data must be broken down from
    # 30 dimensions to `n` dimensions.
    # This is done with Principal Component Analysis (PCA),
    # which finds patterns while keeping variation.
    pca = PCA(n_components=n).fit(X_train)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)

    # The last step in the data processing is
    # to scale the data to be between -1 and 1
    samples = np.append(X_train, X_test, axis=0)
    minmax_scale = MinMaxScaler((-1, 1)).fit(samples)
    X_train = minmax_scale.transform(X_train)
    X_test = minmax_scale.transform(X_test)

    # Now some sample should be picked to train the model from
    training_input = {key: (X_train[Y_train == k, :])[:training_size] for k, key in enumerate(class_labels)}
    test_input = {key: (X_train[Y_train == k, :])[training_size:(
        training_size+test_size)] for k, key in enumerate(class_labels)}

    if PLOT_DATA:
        x_axis_data = X_train[Y_train == 0, 0][:training_size]
        y_axis_data = X_train[Y_train == 0, 1][:training_size]
        
        label = 'Malignant' if 0==1 else 'Benign'
        plt.scatter(x_axis_data, y_axis_data, label=label)
        x_axis_data = X_train[Y_train == 1, 0][:training_size]
        y_axis_data = X_train[Y_train == 1, 1][:training_size]
        
        label = 'Malignant' if 1==1 else 'Benign'
        plt.scatter(x_axis_data, y_axis_data, label=label)

        plt.title("Breast Cancer Dataset (Dimensionality Reduced With PCA)")
        plt.legend()
        plt.show()
        

    return X_train, training_input, test_input, class_labels
