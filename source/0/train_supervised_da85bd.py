# https://github.com/philipkarim/master_thesis/blob/67f537e7fb28b2a67ac664fae7302d0bd4b364f1/src/train_supervised.py
import copy
from dataclasses import replace
import numpy as np
import qiskit as qk
from qiskit.quantum_info import DensityMatrix, partial_trace
import time
import sys
import os

#Import scikit learn modules
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_curve
from sklearn.utils import shuffle

#Import pytorch modules
import torch.optim as optim_torch
import torch

# Import the other classes and functions
from varQITE import *
from utils import *
from optimize_loss import optimize
from NN_class import *

import seaborn as sns

def train_model(dataset, initial_H, ansatz, optim_params, visible_q=1, task='regression', n_steps=10, folder='', network_coeff=None, nickname=None, init_w='xavier_normal'):
    """
    Function to run regression of the franke function with the variational Boltzmann machine

    Args:
            initial_H(array):   The Hamiltonian which will be used, the parameters 
                                will be initialized within this function

            ansatz(array):      Ansatz whill be used in the VarQBM

            network_coeff(list): layerwise [input, output, bias], 0 if no bias, 1 with bias

    Returns:    Scores on how the BM performed
    """
    #Dataset
    X_train=dataset[0]; y_train=dataset[1]; X_test=dataset[2];  y_test=dataset[3]
    #Optimizarion parameters
    n_epochs=optim_params[0];   opt_met=optim_params[1];    lr=optim_params[2]; m1=optim_params[3]; m2=optim_params[4]
    
    visible_q_list=list(range(visible_q))

    #Some default Hamiltonians
    if initial_H==1:
        hamiltonian=[[[0., 'z', 0], [0., 'z', 1]], [[0., 'z', 0]], [[0., 'z', 1]]]
        n_hamilParameters=len(hamiltonian)
    elif initial_H==2:
        hamiltonian=[[[0., 'z', 0], [0., 'z', 1]], [[0., 'z', 0]], [[0., 'z', 1]], [[0.1, 'x', 0]],[[0.1, 'x', 1]]]
        n_hamilParameters=len(hamiltonian)-2
    elif initial_H==3:
        hamiltonian=[[[0., 'z', 0], [0., 'z', 1]], [[0., 'z', 0]], [[0., 'z', 1]], [[0, 'x', 0]],[[0, 'x', 1]]]
        n_hamilParameters=len(hamiltonian)

    else:
        print('Hamiltonian not defined')
        exit()

    
    #Initializing the parameters:
    if network_coeff is not None:
        #Initializing the network
        net=Net(network_coeff, X_train[0], n_hamilParameters)
        
        if init_w=='xavier_normal':
            net.apply(init_weights_XN)

        elif init_w=='xavier_uniform':
            net.apply(init_weights_XU)

        elif init_w=='he_normal':
            net.apply(init_weights_HN)

        elif init_w=='he_uniform':
            net.apply(init_weights_HU)

        elif init_w=='normal':
            net.apply(init_weights_N)

        elif init_w=='uniform':
            net.apply(init_weights_U)

        else:
            sys.exit('Neural network nitialization not known')

        #Floating the network parameters
        net = net.float()

        """
        weights_and_bias=[]
        for name, param in net.named_parameters():
            print(name)
            for i in param.data:
                temp_weight=[]
                #print(len(i))
                try:
                    for j in i:
                        temp_weight.append(np.array(j.item(), dtype=object))
                        print(j.item())
                    weights_and_bias.append(np.array(temp_weight, dtype=object))
                except:
                    pass
                    #temp_weight.append(np.array(i.item()))

        print(np.array(weights_and_bias,dtype=object))
        """
        
        if opt_met=='SGD':
            optimizer = optim_torch.SGD(net.parameters(), lr=lr)
            m1=0; m2=0
        elif opt_met=='Adam':
            optimizer = optim_torch.Adam(net.parameters(), lr=lr, betas=[m1, m2])
        elif opt_met=='Amsgrad':
            optimizer = optim_torch.Adam(net.parameters(), lr=lr, betas=[m1, m2],amsgrad=True)
        elif opt_met=='RMSprop':
            optimizer = optim_torch.RMSprop(net.parameters(), lr=lr, alpha=m1)
            m2=0
        else:
            print('optimizer not defined')
            exit()

        H_parameters=net(X_train[0])

    else:            
        #Initializing the parameters:
        H_parameters=np.random.uniform(low=-1.0, high=1.0, size=((n_hamilParameters, len(X_train[0]))))
        H_parameters = torch.tensor(H_parameters, dtype=torch.float64, requires_grad=True)

        if opt_met=='SGD':
            optimizer = optim_torch.SGD([H_parameters], lr=lr)
            m1=0; m2=0
        elif opt_met=='Adam':
            optimizer = optim_torch.Adam([H_parameters], lr=lr, betas=[m1, m2])
        elif opt_met=='Amsgrad':
            optimizer = optim_torch.Adam([H_parameters], lr=lr, betas=[m1, m2], amsgrad=True)
        elif opt_met=='RMSprop':
            optimizer = optim_torch.RMSprop([H_parameters], lr=lr, alpha=m1)
            m2=0
        else:
            print('Optimizer not defined')
            exit()
    
    init_params=np.array(copy.deepcopy(ansatz))[:, 1].astype('float')

    tracing_q, rotational_indices=getUtilityParameters(ansatz)
    optim=optimize(H_parameters, rotational_indices, tracing_q,loss_func=task,learning_rate=lr, method=opt_met, fraud=True)
    
    varqite_train=varQITE(hamiltonian, ansatz, steps=n_steps, symmetrix_matrices=False)
    varqite_train.initialize_circuits()

    loss_mean=[]
    loss_mean_test=[]
    H_coefficients=[]
    predictions_train=[]
    targets_train=[]
    predictions_test=[]
    targets_test=[]
    target_score=[]

    for epoch in range(n_epochs):
        start_time=time.time()
        print(f'Epoch: {epoch}/{n_epochs}')

        #Lists to save the predictions of the epoch
        pred_epoch=[]
        loss_list=[]
        targets=[]

        #Loops over each sample
        X_train, y_train = shuffle(X_train, y_train, random_state=0)
        
        for i,sample in enumerate(X_train):
            varqite_time=time.time()
            #Updating the Hamiltonian with the correct parameters
            if network_coeff is not None:
                #Network parameters
                output_coef=net(sample)

                for term_H in range(n_hamilParameters):
                    for qub in range(len(hamiltonian[term_H])):
                        hamiltonian[term_H][qub][0]=output_coef[term_H]
            else:
                for term_H in range(n_hamilParameters):
                    for qub in range(len(hamiltonian[term_H])):
                        hamiltonian[term_H][qub][0]=bias_param(sample, H_parameters[term_H])

            #Updating the Hamiltonian
            varqite_train.update_H(hamiltonian)
            ansatz=update_parameters(ansatz, init_params)
            omega, d_omega=varqite_train.state_prep(gradient_stateprep=False)
            ansatz=update_parameters(ansatz, omega)
            trace_circ=create_initialstate(ansatz)

            DM=DensityMatrix.from_instruction(trace_circ)
            PT=partial_trace(DM,tracing_q)
            p_QBM = PT.probabilities(visible_q_list)
            
            target_data=np.zeros(2**visible_q)
            if task=='classification':
                target_data[y_train[i]]=1
                loss=optim.cross_entropy(target_data,p_QBM)
                if visible_q==1:
                    pred_epoch.append(0) if p_QBM[0]>0.5 else pred_epoch.append(1)
                else:
                    pred_epoch.append(np.where(p_QBM==p_QBM.max())[0][0])
                
                targets.append(target_data)

            elif task=='regression':
                loss=optim.MSE(y_train[i],p_QBM[0])
                target_data[0]=y_train[i]; target_data[1]=1-target_data[0]
                pred_epoch.append(p_QBM[0])
                targets.append(target_data[0])

            else:
                sys.exit('Task not defined (classification/regression)')

            print(f'TRAIN: Loss: {loss}, p_QBM: {p_QBM}, target: {target_data}')

            #Add scores and predictions to lists
            loss_list.append(loss)

            gradient_qbm=optim.fraud_grad_ps(hamiltonian, ansatz, d_omega, visible_q_list)
            gradient_loss=optim.gradient_loss(target_data, p_QBM, gradient_qbm)

            #print(f'Ham before optimizer: {hamiltonian}')
            optimizer.zero_grad()
            if network_coeff is not None:
                output_coef.backward(torch.tensor(gradient_loss, dtype=torch.float64))
            else:
                gradient=np.zeros((len(gradient_loss),len(sample)))
                for ii, grad in enumerate(gradient_loss):
                    for jj, samp in enumerate (sample):
                        gradient[ii][jj]=grad*samp
                
                H_parameters.backward(torch.tensor(gradient, dtype=torch.float64))

            optimizer.step()

            print(f'1 sample run train: {time.time()-varqite_time}')

        
        #Computes the test scores regarding the test set:
        loss_mean.append(np.mean(loss_list))
        predictions_train.append(pred_epoch)
        targets_train.append(targets)

        #H_coefficients.append(np.array(torch.clone(H_parameters).detach()))

        temp=[]
        #print(H_parameters)
        if network_coeff is not None:
            for i in range(len(output_coef)):            
                temp.append(output_coef[i].item())
        else:
            for i in H_parameters:
                temp2=[]
                for j in i:
                    temp2.append(j.item())
                temp.append(np.array(temp2))
        H_coefficients.append(np.array(temp))

        #print(H_coefficients)

        #np.save('H_coeff_letest.npy', np.array(H_coefficients))


        print(f'Train Epoch complete : mean loss list= {loss_mean}')

        #Testing stage
        loss_list=[]
        pred_epoch=[]
        targets=[]
        #target scores are the probability given for the true label
        target_score_epoch=[]

        with torch.no_grad():
            for i,sample in enumerate(X_test):
                test_samp_time=time.time()
                if network_coeff is not None:
                    #Network parameters
                    output_coef=net(sample)
                    for term_H in range(n_hamilParameters):
                        for qub in range(len(hamiltonian[term_H])):
                            hamiltonian[term_H][qub][0]=output_coef[term_H]
                else:
                    for term_H in range(n_hamilParameters):
                        for qub in range(len(hamiltonian[term_H])):
                            hamiltonian[term_H][qub][0]=bias_param(sample, H_parameters[term_H])

                #Updating the Hamiltonian
                varqite_train.update_H(hamiltonian)
                ansatz=update_parameters(ansatz, init_params)
                omega, not_used=varqite_train.state_prep(gradient_stateprep=True)
                ansatz=update_parameters(ansatz, omega)
                trace_circ=create_initialstate(ansatz)

                DM=DensityMatrix.from_instruction(trace_circ)
                PT=partial_trace(DM,tracing_q)
                p_QBM = PT.probabilities(visible_q_list)

                target_data=np.zeros(2**visible_q)
                if task=='classification':
                    target_data[y_test[i]]=1
                    loss=optim.cross_entropy(target_data,p_QBM)
                    target_score_epoch.append(p_QBM[y_test[i]])

                    if visible_q==1:
                        pred_epoch.append(0) if p_QBM[0]>0.5 else pred_epoch.append(1)
                    else:
                        pred_epoch.append(np.where(p_QBM==p_QBM.max())[0][0])
                    
                    targets.append(target_data)

                elif task=='regression':
                    loss=optim.MSE(y_train[i],p_QBM[0])
                    target_data[0]=y_train[i]; target_data[1]=1-target_data[0]
                    pred_epoch.append(p_QBM[0])
                    targets.append(target_data[0])

                else:
                    sys.exit('Task not defined (classification/regression)')

                loss_list.append(loss)

                print(f'Test sample time: {time.time()-test_samp_time}')
                
            #Computes the test scores regarding the test set:
            loss_mean_test.append(np.mean(loss_list))
            predictions_test.append(pred_epoch)
            targets_test.append(targets)
            if task=='classification':
                target_score.append(np.array(target_score_epoch))
            print(f'TEST: Loss: {loss_mean_test[-1],loss_mean_test}')
        
        print(f'Time of epoch: {time.time()-start_time}')
    
    del optim
    del varqite_train

    #Save the scores
    if nickname is not None:
        path='results/disc_learning/'+folder
        dir_exist = os.path.exists('results/disc_learning/'+folder)

        if not dir_exist:
            # Create a new directory because it does not exist
            os.makedirs(path)

        np.save(path+'/loss_test'+nickname+'.npy', np.array(loss_mean_test))
        np.save(path+'/loss_train'+nickname+'.npy', np.array(loss_mean))
        np.save(path+'/predictions_train'+nickname+'.npy', np.array(predictions_train))
        np.save(path+'/predictions_test'+nickname+'.npy', np.array(predictions_test))
        np.save(path+'/targets_train'+nickname+'.npy', np.array(targets_train))
        np.save(path+'/targets_test'+nickname+'.npy', np.array(targets_test))
        np.save(path+'/H_coeff'+nickname+'.npy', np.array(H_coefficients))
        if task=='classification':
                np.save(path+'/targets_test_score'+nickname+'.npy', np.array(target_score))


