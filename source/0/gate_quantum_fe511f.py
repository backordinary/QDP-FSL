# https://github.com/vinitX/RBM/blob/8f43268496a5ad0ca8adabe4e793627e9556a96c/gate_quantum.py
#!/usr/bin/env python
# coding: utf-8
from gradient import *
import qiskit
import itertools
import numpy as np
from qiskit import IBMQ
from qiskit import *
from qiskit.quantum_info import Statevector
from collections import Counter
import scipy.sparse
from qiskit import IBMQ, assemble, transpile
from qiskit.ignis.mitigation.measurement import (complete_meas_cal,CompleteMeasFitter)

np.load.__defaults__=(None, True, True, 'ASCII')



def calculate_gate(para1, n, m):
    a,b,w,v,c,e,f = para1
    k = max(np.sum(np.abs(w))/2,1)
    qr = QuantumRegister(m+n+m*n, name='qr')     #m*n ancillas for C-C-Ry action
    cr = ClassicalRegister(m+n+m*n, name='cr')    #m*n classical bits for storing measurement results of ancilla
    
    circuit = QuantumCircuit(qr, cr)

    # Function to compute rotation angle from weights and biases
    #************************************************************
    def calculate_rotation(para, n, m):        
        theta = []
        gamma = []
        theta_weights_positive = []
        theta_weights_negative = []
        
        
        for i in range(len(a)):
            pro = np.sqrt(np.exp(a[i][0])/(np.exp(a[i][0]) + np.exp(-a[i][0])))
            theta1 = np.arcsin(pro)*2
            theta.append(theta1)
            
            
        for i in range(len(b)):
            pro = np.sqrt(np.exp(b[i][0])/(np.exp(b[i][0]) + np.exp(-b[i][0])))
            gamma1 = np.arcsin(pro)*2
            gamma.append(gamma1)
            
        for i in range(n):
            for j in range(m):
                weight = w[i][j]
                weight_bound = np.abs(w[i][j])
                pro_weights_positive = np.sqrt(np.exp(weight)/(np.exp(weight_bound)))
                theta_weights_positive1 = np.arcsin(pro_weights_positive)*2
                theta_weights_positive.append(theta_weights_positive1)
                
        for i in range(n):
            for j in range(m):
                weight = w[i][j]
                weight_bound = np.abs(w[i][j])
                pro_weights_negative = np.sqrt(np.exp(-weight)/(np.exp(weight_bound)))
                theta_weights_negative1 = np.arcsin(pro_weights_negative)*2
                theta_weights_negative.append(theta_weights_negative1)
            
        return theta, gamma, theta_weights_positive, theta_weights_negative

    theta, gamma, theta_weights_positive, theta_weights_negative = calculate_rotation(para1, n, m)
    #print (theta_weights_positive, theta_weights_negative)
    
    # Ry gates for biases of vis and hidden layers
    #***********************************************
    for item in np.arange(m+n): 
        if (item < n):
            circuit.ry(theta[item],item)   # only on vis layer
        else: 
            circuit.ry(gamma[item-n],item)  # only on hidden layer

    
    # C-C-Ry gates for weights between hidden and vis layers
    #*********************************************************

    for qubit_pair in list(itertools.product(np.arange(n), np.arange(m))):
        pair_index = list(itertools.product(np.arange(n), np.arange(m))).index(qubit_pair)        
        vis_qubit =qubit_pair[0]
        hid_qubit=qubit_pair[1]
    #For 11 state
        circuit.mcry(theta_weights_positive[pair_index],[qr[vis_qubit],qr[n+hid_qubit]],qr[m+n+pair_index],None, mode="noancilla")
    
    #For 01 state
        circuit.x(vis_qubit)
        circuit.mcry(theta_weights_negative[pair_index],[qr[vis_qubit],qr[n+hid_qubit]],qr[m+n+pair_index],None, mode="noancilla")
        circuit.x(vis_qubit)

    #For 10 state
        circuit.x(n+hid_qubit)
        circuit.mcry(theta_weights_negative[pair_index],[qr[vis_qubit],qr[n+hid_qubit]],qr[m+n+pair_index],None, mode="noancilla")
        circuit.x(n+hid_qubit)
    #For 00 state
        circuit.x(vis_qubit)
        circuit.x(n+hid_qubit)
        circuit.mcry(theta_weights_positive[pair_index],[qr[vis_qubit],qr[n+hid_qubit]],qr[m+n+pair_index],None, mode="noancilla")
        circuit.x(n+hid_qubit)
        circuit.x(vis_qubit)
    
    #For qubits 0 and 2

    #For 11 state
    #circuit.mcry(theta_weights_positive[0],[qr[0],qr[1]],qr[2],None, mode="noancilla")
    
    #For 01 state
    #circuit.x(0)
    #circuit.mcry(theta_weights_negative[0],[qr[0],qr[1]],qr[2],None, mode="noancilla")
    #circuit.x(0)

    #For 10 state
    #circuit.x(1)
    #circuit.mcry(theta_weights_negative[0],[qr[0],qr[1]],qr[2],None, mode="noancilla")
    #circuit.x(1)
    
    #For 00 state
    #circuit.x(0)
    #circuit.x(1)
    #circuit.mcry(theta_weights_positive[0],[qr[0],qr[1]],qr[2],None, mode="noancilla")
    #circuit.x(1)
    #circuit.x(0)
    
        
        
    circuit.measure(qr,cr)
    

    simulator = Aer.get_backend('qasm_simulator')
    result = execute(circuit, backend = simulator, shots=1000000).result()
    
    try:
        my_result = result.get_counts(circuit)
    except QiskitError:
        print(result.error_message())
    
#    print("Statevector = ", statevector.shape)
#    my_result = Statevector.to_counts(my_result)
    
    my_filter = dict()
    ancilla_string='1'*(m*n)
    for i in range(2**(m+n)):
        #print (bin_transform(i,m,n))
        #print ('1'+ bin_transform(i,m,n)[::-1])
        
        my_filter[i] = my_result.get((ancilla_string + bin_transform(i,m,n)[::-1]), 1e-16)
        
   
    dict_values = my_filter.values()
    sum_values = sum(dict_values)
#    print("sum counts = ", sum_values)

    probabilities = []

    for key in my_filter:
        value = my_filter[key]
        prob = np.divide((value),(sum_values))
        probabilities.append(prob)
        
    state = np.array(probabilities)
    state_p = ((state)/(sum(probabilities))).reshape(2**n,2**m)

    state = state_p**k/np.sum(state_p**k)

    return(state.reshape(2**n,2**m), k) 



