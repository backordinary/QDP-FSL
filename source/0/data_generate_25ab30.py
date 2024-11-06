# https://github.com/Zao-0/Figure_Phase_Transition_with_CNN/blob/2d263829cad2f77d36ece982f10c5bf66255a298/data_generate.py
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 17:39:54 2022

@author: Zao
"""

import scipy.sparse as spa
import numpy as np
#import time
import torch
import os
import qiskit
from tqdm import tqdm


# define 3 matrix for quation a
Id = spa.csr_matrix(np.array(([1,0], [0,1])))
Sx = spa.csr_matrix(np.array(([0,1], [1,0])))
Sz = spa.csr_matrix(np.array(([1,0], [0,-1])))
Sy = spa.csr_matrix(np.array(([0,-1j],[1j,0])))
Ini = spa.csr_matrix(np.array([1]))

def sigma_matrix(L,base):
    l = []
    for i in range(L):
        result = Ini
        for j in range(i):
            result = spa.kron(result, Id, format = 'csr')
        if base=='Sx':
            result = spa.kron(result, Sx, format = 'csr')
        elif base=='Sz':
            result = spa.kron(result, Sz, format = 'csr')
        elif base=='Sy':
            result = spa.kron(result, Sy, format = 'csr')
        else:
            result = spa.korn(result, Id, format = 'csr')
        for j in range(L-i-1):
            result = spa.kron(result, Id, format = 'csr')
        l.append(result)
    return l

def gen_hamiltonian(sz_list, sx_list, sy_list, g_list,J):
    H = -g_list[0]*sz_list[0]
    for i in range(1, len(sz_list)):
        H -= g_list[i]*sz_list[i]
    for i in range(len(sx_list)):
        H-=J*sx_list[i]*sx_list[(i+1)%len(sx_list)]
        H-=J*sy_list[i]*sy_list[(i+1)%len(sx_list)]
        H-=J*sz_list[i]*sz_list[(i+1)%len(sx_list)]
    return spa.csr_matrix(H)

def gen_h_value(W, size):
    h_arr = np.random.uniform(-W,W, size)
    return h_arr

def get_trace_subspace(L,n):
    assert n>L/4 and n<L
    l = []
    for i in range(L-n+1):
        c_l = list(range(L))
        for j in range(n):
            c_l.pop(i)
        l.append(c_l)
    return l

def data_generator(L, n, J, w_index,tag, sample = 5000, eps = 0.02):
    h_arr = gen_h_value(w_index*J, L)
    subspace_list = get_trace_subspace(L, n)
    index = 0
    fld = 'D:\\pic\\train\\n_{}\\tag_{}\\'.format(n,tag)
    sz_list = sigma_matrix(L, 'Sz')
    sx_list = sigma_matrix(L, 'Sx')
    sy_list = sigma_matrix(L, 'Sy')
    print(os.path.exists(fld))
    if not os.path.exists(fld):
        os.makedirs(fld)
    #c=input()
    while index<sample:
        H = gen_hamiltonian(sz_list, sx_list, sy_list,h_arr, J)
        egv, egvect = np.linalg.eigh(H.toarray())
        a = egv.shape[0]
        vect_list = []
        for i in range(a):
            if abs(egv[i])<eps:
                vect_list.append(egvect[:,i])
        #print(len(vect_list))
        for vect in vect_list:
            for sub in subspace_list:
                ts = np.array(qiskit.quantum_info.partial_trace(vect, sub))
                ts = ts.real
                ts = ts/np.linalg.norm(ts)
                ts = torch.tensor(ts)
                #print('save tensor')
                torch.save(ts, fld+'{}.pt'.format(index))
                #print('finish saving')
                #c = input()
                index+=1

np.random.seed(2264)
L = 12
J = 1
for n in [7,8]:
    for tag in [0,1]:
        w_index = 0.5
        if tag:
            w_index = 8
        data_generator(L, n, J, w_index,tag)
