# https://github.com/vinitX/RBM/blob/8f43268496a5ad0ca8adabe4e793627e9556a96c/gradient.py
from gate_quantum import *
import sys
import csv
import qiskit
import numpy as np
from qiskit import IBMQ
from qiskit import *
from qiskit.quantum_info import Statevector
from collections import Counter
import scipy.sparse
from qiskit import IBMQ, assemble, transpile
from qiskit.ignis.mitigation.measurement import (complete_meas_cal,CompleteMeasFitter)

np.load.__defaults__=(None, True, True, 'ASCII')


def back(state_p,state_sign,para,state,Hamiltonian):
    # this is for backward procedure
    # we have to specific the dim of all value
    
    a,b,w,v,c,e,f=para
    
    state_v, state_e, state_h, state_1=state

#    print("State sign = ", state_sign)
    #prob(2^n_v,1)
    prob1=np.sum(state_p,axis=1,keepdims=True)/np.sum(state_p)

    #prob = sampling((np.sum(state_p,axis=1,keepdims=True)/np.sum(state_p)).reshape(2**len(b)),10000)

    phi = np.sqrt(prob1)*state_sign

    prob = phi * np.conjugate(phi)
    

    da_grad1 = np.sum((0.5*state_v*(np.dot(Hamiltonian,phi)/phi)*prob), axis=0,keepdims=True)/np.sum(prob)  + np.sum((0.5*state_v*(np.conjugate(np.dot(Hamiltonian,phi)/phi))*prob), axis=0,keepdims=True)/np.sum(prob) -  np.sum(state_v*(np.sum(((np.dot(Hamiltonian,phi)/phi)*prob))/np.sum(prob))*prob, axis=0,keepdims=True)/np.sum(prob)

    da_grad = da_grad1.reshape((a.shape[0],1))
    da = np.real(da_grad)

    
    
    db_grad1 = np.sum((0.5*(np.tanh(np.dot(w.T,state_v.T)+b))*(np.dot(Hamiltonian,phi)/phi).T*prob.T), axis=1,keepdims=True).T/np.sum(prob)  + np.sum((0.5*(np.tanh(np.dot(w.T,state_v.T)+b))*(np.conjugate(np.dot(Hamiltonian,phi)/phi)).T*prob.T), axis=1,keepdims=True).T/np.sum(prob) -  np.sum((np.tanh(np.dot(w.T,state_v.T)+b))*(np.sum(((np.dot(Hamiltonian,phi)/phi).T*prob.T))/np.sum(prob))*prob.T, axis=1,keepdims=True).T/np.sum(prob)

    db_grad = db_grad1.reshape((b.shape[0],1))

    db = np.real(db_grad)

    

    dw_grad1 = np.dot((0.5*state_v.T*(np.dot(Hamiltonian,phi)/phi).T*prob.T), (np.tanh(np.dot(w.T,state_v.T)+b).T))/np.sum(prob) + np.dot((0.5*state_v.T*(np.conjugate(np.dot(Hamiltonian,phi)/phi)).T*prob.T), (np.tanh(np.dot(w.T,state_v.T)+b).T))/np.sum(prob) -  np.dot((state_v.T*(np.sum(((np.dot(Hamiltonian,phi)/phi).T*prob.T))/np.sum(prob))*prob.T), (np.tanh(np.dot(w.T,state_v.T)+b).T))/np.sum(prob)

    dw_grad = dw_grad1.reshape((w.shape[0],w.shape[1]))

    dw = np.real(dw_grad) 

    
    
    dc_grad = np.sum(((np.dot(Hamiltonian,phi)/phi)*prob)*np.conjugate(1/state_sign -state_sign))/np.sum(prob) + np.sum((((np.conjugate(np.dot(Hamiltonian,phi)/phi))*prob)*(1/state_sign -state_sign)))/np.sum(prob) -  (np.sum(((np.sum(((np.dot(Hamiltonian,phi)/phi)*prob))/np.sum(prob))*prob*(np.conjugate(1/state_sign -state_sign))) + (np.sum((np.conjugate(np.dot(Hamiltonian,phi)/phi)*prob))/np.sum(prob))*prob*((1/state_sign -state_sign))))/np.sum(prob)


    dc = np.real(dc_grad)

    
    dv_grad1 = np.sum((state_v*(np.dot(Hamiltonian,phi)/phi)*prob)*np.conjugate(1/state_sign -state_sign), axis=0)/np.sum(prob) + np.sum((state_v*((np.conjugate(np.dot(Hamiltonian,phi)/phi))*prob)*(1/state_sign -state_sign)), axis=0)/np.sum(prob) -  (np.sum((state_v*(np.sum(((np.dot(Hamiltonian,phi)/phi)*prob))/np.sum(prob))*prob*(np.conjugate(1/state_sign -state_sign)))+ (state_v*(np.sum((np.conjugate(np.dot(Hamiltonian,phi)/phi)*prob))/np.sum(prob))*prob*((1/state_sign -state_sign))), axis=0))/np.sum(prob)

    dv_grad = dv_grad1.reshape((v.shape[0],1)) 

    dv = np.real(dv_grad) 


    de_grad = np.sum(((np.dot(Hamiltonian,phi)/phi)*prob)*np.conjugate((1j)*(1/state_sign -state_sign)))/np.sum(prob) + np.sum((((np.conjugate(np.dot(Hamiltonian,phi)/phi))*prob)*((1j)*(1/state_sign -state_sign))))/np.sum(prob) -  (np.sum(((np.sum(((np.dot(Hamiltonian,phi)/phi)*prob))/np.sum(prob))*prob*(np.conjugate((1j)*(1/state_sign -state_sign)))) + (np.sum((np.conjugate(np.dot(Hamiltonian,phi)/phi)*prob))/np.sum(prob))*prob*((1j)*(1/state_sign -state_sign))))/np.sum(prob)

    de = np.real(de_grad)


    
    df_grad1 = np.sum((state_v*(np.dot(Hamiltonian,phi)/phi)*prob)*np.conjugate((1j)*(1/state_sign -state_sign)), axis=0)/np.sum(prob) + np.sum((state_v*((np.conjugate(np.dot(Hamiltonian,phi)/phi))*prob)*(1j)*(1/state_sign -state_sign)), axis=0)/np.sum(prob) -  (np.sum((state_v*(np.sum(((np.dot(Hamiltonian,phi)/phi)*prob))/np.sum(prob))*prob*(np.conjugate((1j)*(1/state_sign -state_sign))))                 + (state_v*(np.sum((np.conjugate(np.dot(Hamiltonian,phi)/phi)*prob))/np.sum(prob))*prob*((1j)*(1/state_sign -state_sign))), axis=0))/np.sum(prob)

    df_grad = df_grad1.reshape((f.shape[0],1)) 

    df = np.real(df_grad)

 
    return (da,db,dw,dv,dc,de,df)


def initial_para(n_v, n_h, rate):
    
    a = (2*np.random.rand(n_v,1)-1)*rate
    b = (2*np.random.rand(n_h,1)-1)*rate
    w = (2*np.random.rand(n_v,n_h)-1)*rate
    v = (2*np.random.rand(n_v,1)-1)*rate
    c = (2*np.random.rand()-1)*rate
    f = (2*np.random.rand(n_v,1)-1)*rate
    e = (2*np.random.rand()-1)*rate
    
    return (a,b,w,v,c,e,f)


def transfer_para(n_v, n_h, rate, index):

#    np_load_old = np.load
#    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
    t_para = np.load('parameters'+str(index)+'_exc.npy')
#    np.load = np_load_old
    a,b,w,v,c,e,f = t_para
    
    return (a,b,w,v,c,e,f)    



def bin_transform(i,m,n):
   
    if i >= 2**(m+n):
        return False
   
    string = str(bin(i))[2:]
   
    for j in range((m+n)-len(string)):
       
        string = '0' + string

    return string



def update(para, update_value,learning_rate):
    
    a,b,w,v,c,e,f = para
    da,db,dw,dv,dc,de,df = update_value
    a=a-learning_rate*(da)
    b=b-learning_rate*(db)
    w=w-learning_rate*(dw)
    v=v-learning_rate*(dv)#+np.random.rand(len(db),1)*0.01)
    c=c-learning_rate*dc
    e=e-learning_rate*de
    f=f-learning_rate*(df)
    
    return (a,b,w,v,c,e,f)


def sign_prog(state,para):
    
    #this is to calculate the sign
    
    a,b,w,v,c,e,f=para
    
    state_v, state_e, state_h, state_p=state

    state_p=np.zeros((2**len(a),1), dtype = np.complex128)
    
    complex_sign = (np.sum((state_v*v.T),axis=1,keepdims=True)+c) + (1j)*(np.sum((state_e*f.T),axis=1,keepdims=True)+e)

    state_p+= complex_sign
    
    #state_sign=np.cos(1*state_p)

    state_sign = np.tanh(state_p)
    
    return state_sign


def connect(num,array):
    return np.concatenate((np.zeros(num-len(array)),array))


def permutation(num):
    # this is to generate all possible permutation 
    per=np.zeros((2**num,num))
    
    for i in range(2**num):
        per[i]=-np.power(-1,connect(num,np.array(list(map(int,list(bin(i)[2:]))))))
        
    return per    

def initial_state(n_v,n_h):
    #this is to initial the state to RBM with n_v as number of visible and n_h as number of hidden
    
    state_v=permutation(n_v)
    state_e=permutation(n_v)
    state_h=permutation(n_h)
    state_p=np.zeros((2**n_v,2**n_h))
    
    return (state_v,state_e,state_h,state_p)

def prog(state, para):
    # this is for the prog procedure
    # which is used to validate quantum is correct
    
    a,b,w,v,c,e,f = para
    
    #sumall = np.sum(np.abs(b))+np.sum(np.abs(d))+np.sum(np.abs(w))
    
    state_v, state_e, state_h, state_p=state

    state_p=np.zeros((2**len(a),2**len(b)), dtype = np.complex128)
    
    state_p+=np.sum((state_v*a.T),axis=1,keepdims=True)
    
    state_p+=np.sum((state_h*b.T),axis=1,keepdims=True).T
    
    value=w*np.einsum('ij,kl->ikjl', state_v, state_h)
    value=np.sum(value[:,:,i,j] for i in range(len(a)) for j in range(len(b)))
            
    state_p+=value
    
    
    state_p=np.exp(state_p)
    
    
    return (state_p,np.sum(np.abs(w)))


