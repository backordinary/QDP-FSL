# https://github.com/philipkarim/master_thesis/blob/67f537e7fb28b2a67ac664fae7302d0bd4b364f1/src/fraud_classification.py
"""
alt+z to fix word wrap

Rotating the monitor:
xrandr --output DP-1 --rotate right
xrandr --output DP-1 --rotate normal

xrandr --query to find the name of the monitors
"""
import copy
from dataclasses import replace
from operator import truediv
import numpy as np
import qiskit as qk
from qiskit.quantum_info import DensityMatrix, partial_trace
import time



#Import scikit learn modules
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_curve
from sklearn.utils import shuffle

#Import pytorch modules
import torch.optim as optim_torch
import torch
from BM import *
from ml_methods_class import MlMethods

# Import the other classes and functions
from varQITE import *
from utils import *
from optimize_loss import optimize
from NN_class import *
from train_supervised import train_model

import seaborn as sns

#sns.set_style("darkgrid")

def fraud_detection(H_num, ansatz, n_epochs, lr, opt_met, m1=0.99, m2=0.99, v_q=1, layers=None, ml_task='classification', directory='fraud', name=None, init_ww='xavier_normal', QBM=True, samp_400=False):
    """
    Function to run fraud classification with the variational Boltzmann machine

    Args:
            initial_H(array):   The Hamiltonian which will be used, the parameters
                                will be initialized within this function

            ansatz(array):      Ansatz whill be used in the VarQBM

            network_coeff(list): layerwise [input, output, bias], 0 if no bias, 1 with bias

    Returns:    Scores on how the BM performed
    """
    #Importing the data
    fraud_20=True
    #Ensures 400 samples
    if fraud_20==True:
        dataset_fraud=np.load('datasets/time_amount_zip_mcc_1000_instances.npy', allow_pickle=True)
        #Start by normalizing the dataset by subtracting the mean and dividing by the deviation:

        #Split the data into X and y
        X=np.hsplit(dataset_fraud, (len(dataset_fraud[0])-1,len(dataset_fraud[0])))
        y=X[1].astype('int')
        X=X[0]

        ## The reason we split it like this instead of the regular train_test_split is to secure
        ## The right amount of true samples in each set(Probably is a lot more efficient way to do this)

        #Extracts indices of samples which are fraud and not fraud
        true_indices=np.where(y==1)[0]
        false_indices=np.where(y==0)[0]

        
        #Train: 20% 100/500
        #train_true_samples=100
        #test_true_samples=50
        #train_false_samples=400
        #test_false_samples=200
        
        train_true_samples=80
        test_true_samples=25
        train_false_samples=320
        test_false_samples=225

        #Parameter search 100 samples
        #train_true_samples=20
        #test_true_samples=10
        #train_false_samples=80
        #test_false_samples=40

        #Makes sure that each set of data contains the wanted number of true samples
        train_indices=np.random.choice(true_indices, train_true_samples, replace=False)
        true_indices = np.delete(true_indices, np.where(np.in1d(true_indices, train_indices)))
        test_indices=np.random.choice(true_indices, test_true_samples, replace=False)
        val_indices = np.delete(true_indices, np.where(np.in1d(true_indices, test_indices)))

        #Random sampling from the false samples
        train_indices_false=np.random.choice(false_indices, train_false_samples, replace=False)
        false_indices = np.delete(false_indices, np.where(np.in1d(false_indices, train_indices_false)))
        test_indices_false=np.random.choice(false_indices, test_false_samples, replace=False)
        val_indices_false = np.delete(false_indices, np.where(np.in1d(false_indices, test_indices_false)))

        y_train_indices=np.sort(np.concatenate((train_indices, train_indices_false)))
        y_test_indices=np.sort(np.concatenate((test_indices, test_indices_false)))
        y_val_indices=np.sort(np.concatenate((val_indices, val_indices_false)))

        #Splits the samples according to the sampling of y
        X_train=X[y_train_indices]
        X_test=X[y_test_indices]
        X_val=X[y_val_indices]

        y_train=y[y_train_indices]
        y_test=y[y_test_indices]
        y_val=y[y_val_indices]

        if samp_400:
            X_train=X_train[:400]; y_train=y_train[:400]

    else:
        dataset_fraud=np.load('datasets/time_amount_zip_mcc_1000_instances_5050.npy', allow_pickle=True)

        X=np.hsplit(dataset_fraud, (len(dataset_fraud[0])-1,len(dataset_fraud[0])))
        y=X[1].astype('int')
        X=X[0]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    if QBM==False:
        scaler=MinMaxScaler()
    else:
        scaler=StandardScaler()

    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    if fraud_20==True:
        X_val = scaler.transform(X_val)

 
    y_test=np.ravel(y_test)
    y_train=np.ravel(y_train)

    binar=False
    plot_cm=True
    acc=True
    n_epoc=20

    data_fraud=[X_train, y_train, X_test, y_test]
    params_fraud=[n_epochs, opt_met, lr, m1, m2]

    if QBM==True:
        train_model(data_fraud, H_num, ansatz, params_fraud, visible_q=v_q, task=ml_task, folder=directory, network_coeff=layers, nickname=name, init_w=init_ww)
    elif QBM==False:
        test_data=[X_val, y_val]
        best_params=None
        #best_params=gridsearch_params(data_fraud, 20, binarize_data=binar)
        train_rbm(data_fraud, best_params, plot_acc_vs_epoch=n_epoc, name='fraud', binarize_data=binar, plot_acc=acc, cm=plot_cm, data_val=test_data)
        #rbm_plot_scores(data_fraud, name='fraud2', binarize_input=binar)
    else:
        model=MlMethods(data_fraud[0], data_fraud[1], data_fraud[2], data_fraud[3])
        
        if QBM=='NN':
            model.neural_net(1, lr, network_coeff=NN_nodes(8,5, sig_last=True))
        
        elif QBM=='OLS':
            model.ols_reg()
        
        elif QBM=='Ridge':
            model.ridge_reg()
       
        elif QBM=='Lasso':
            model.lasso_reg()
        
        elif QBM=='KNN':
            model.k_nn()
        
        elif QBM=='logreg':
            model.logistic_reg()
        
        else:
            sys.exit('No method defined')
            