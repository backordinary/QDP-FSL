# https://github.com/foxwy/QELM-VQELM/blob/3178820091cebf19d001937afac7827f98c545fe/VQELM_classification/VQELM_UCI.py
# -*- coding: utf-8 -*-
# @Author: foxwy
# @Date:   2020-06-02 15:58:09
# @Last Modified by:   WY
# @Last Modified time: 2021-07-29 18:49:15

import os
import sys
import random
import numpy as np
import copy
import matplotlib.pyplot as plt
import scipy
from scipy.linalg import expm
import math
from sklearn.metrics import log_loss
from openpyxl import Workbook
from timeit import default_timer as timer
import multiprocessing as mp

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer, IBMQ
from qiskit.providers.ibmq.managed import IBMQJobManager
from qiskit.providers.ibmq import least_busy
from qiskit.algorithms.optimizers import NELDER_MEAD, NFT, COBYLA, SPSA, SLSQP, AQGD, TNC,\
                                         ADAM, P_BFGS, CG, POWELL, QNSPSA, GradientDescent\

# -----external libraries-----
sys.path.append('..')

from basic_function import elm
from basic_function import dataset
from basic_function.dataset import ACC

# -----backend-----
backend = Aer.get_backend('qasm_simulator')
'''
IBMQ.load_account()
provider = IBMQ.get_provider(group='open')
backend = provider.get_backend('ibmq_qasm_simulator')'''

'''
provider = IBMQ.get_provider(group='open')
small_devices = provider.backends(filters=lambda x: x.configuration().n_qubits == 5
                                   and not x.configuration().simulator)
backend = least_busy(small_devices)'''
print(backend)

# ----------function----------
plt.style.use(['science', 'no-latex'])
plt.rcParams["font.family"] = 'Arial'

font_size = 23  # 34, 28, 38
font = {'size': font_size, 'weight': 'normal'}

colors = ['#75bbfd', 'magenta', '#658b38', '#c79fef', '#06c2ac', 'cyan', '#ff9408', '#430541', 'blue', '#fb2943']

def Plt_set(ax, xlabel, ylabel, title, loc=2, ncol=1, f_size=17):
    ax.tick_params(labelsize=font_size)
    ax.spines['bottom'].set_linewidth(3)
    ax.spines['left'].set_linewidth(3)
    ax.spines['right'].set_linewidth(3)
    ax.spines['top'].set_linewidth(3)
    ax.tick_params(width=3)
    font2 = {'size': f_size, 'weight': 'normal'}
    ax.legend(prop=font2, loc=loc, frameon=True, ncol=ncol)
    ax.set_xlabel(xlabel, font)  # fontweight='bold'
    ax.set_ylabel(ylabel, font)
    ax.set_title(title, font)


def del_file(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            del_file(c_path)
        else:
            os.remove(c_path)


def Random_unitary_classical(random_theta, nqubit):
    U = 1.0
    for i in range(nqubit):
        U = np.kron(U, np.array([[np.cos(random_theta[i]/2), -np.sin(random_theta[i]/2)], [np.sin(random_theta[i]/2), np.cos(random_theta[i]/2)]]))

    return U


def Unitary_cosin(X, m, a):
    U_cos = 1.0
    n = int(math.log(m, 2))
    m_n_int = round(m / n)
    for i in range(n - 1):
        theta = sum(X[i * m_n_int:(i + 1) * m_n_int]**201) / m_n_int * pi / a
        #theta = math.acos(theta)
        U_cos = np.kron(U_cos, np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]))
    theta = sum(X[(n-1) * m_n_int:]**201) / (m - (n - 1) * m_n_int) * pi / a
    #theta = math.acos(theta)
    U_cos = np.kron(U_cos, np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]))

    return U_cos


def Get_H(x, U):
    return 1.0 / (1 + np.exp(-1.0 * x.dot(U)))


def Get_belta(H, y):
    return scipy.linalg.pinv(H.T.dot(H)).dot(H.T.dot(y))


#--------------------------------------------------------------
# global random unitary
def Random_unitary(random_theta, nqubit):
    qc_U = QuantumCircuit(nqubit)
    for i in range(nqubit):
        qc_U.u(random_theta[i], 0, 0, i)

    return qc_U


# nonlinear mapping unitary
def Unitary_nonlinear(X, nqubit, param):
    qc_cosin = QuantumCircuit(nqubit)
    X = np.array(X)
    m = 2**nqubit
    m_n_int = round(m / nqubit)
    for i in range(nqubit - 1):
        theta = sum(X[i * m_n_int:(i + 1) * m_n_int]**param[1]) / m_n_int * pi / param[0] * 2
        qc_cosin.u(theta, 0, 0, i)
    theta = sum(X[(nqubit-1) * m_n_int:]**param[1]) / (m - (nqubit - 1) * m_n_int) * pi / param[0] * 2
    qc_cosin.u(theta, 0, 0, nqubit-1)

    return qc_cosin


#-----basic circuit-----
def param_circuit_unit(nqubits, entanglement, params):
    qc = QuantumCircuit(nqubits)
    
    for n in range(nqubits):
        qc.ry(params[n], n)
        qc.rz(params[n+nqubits], n)
    qc.barrier()

    if entanglement == 'linear':
        for n in range(nqubits-1):
            qc.cx(n, n+1)
    elif entanglement == 'circular':
        for n in range(nqubits-1):
            qc.cx(n, n+1)
        qc.cx(nqubits-1, 0)
    elif entanglement == 'full':
        for n in range(1, nqubits):
            for j in range(nqubits-n):
                qc.cx(j, j+n)
    qc.barrier()

    return qc

def param_circuit(nqubits, entanglement, params_all):
    qc_all = QuantumCircuit(nqubits)
    params_num = len(params_all)
    reps = int(params_num / (nqubits * 2))
    for i in range(reps):
        params = params_all[i * 2 * nqubits: (i + 1) * 2 * nqubits]
        qc = param_circuit_unit(nqubits, entanglement, params)
        qc_all.compose(qc, inplace=True)

    return qc_all


def RealAmplitudes_circuit_unit(nqubits, entanglement, params):
    qc = QuantumCircuit(nqubits)
    
    for n in range(nqubits):
        qc.ry(params[n], n)
    qc.barrier()

    if entanglement == 'linear':
        for n in range(nqubits-1):
            qc.cx(n, n+1)
    elif entanglement == 'circular':
        for n in range(nqubits-1):
            qc.cx(n, n+1)
        qc.cx(nqubits-1, 0)
    elif entanglement == 'full':
        for n in range(1, nqubits):
            for j in range(nqubits-n):
                qc.cx(j, j+n)

    qc.barrier()

    return qc

def RealAmplitudes_circuit(nqubits, entanglement, params_all):
    qc_all = QuantumCircuit(nqubits)
    params_num = len(params_all)
    reps = int((params_num - nqubits) / nqubits)
    for i in range(reps):
        params = params_all[i * nqubits: (i + 1) * nqubits]
        qc = RealAmplitudes_circuit_unit(nqubits, entanglement, params)
        qc_all.compose(qc, inplace=True)
    for n in range(nqubits):
        qc_all.ry(params_all[reps * nqubits:][n], n)
    qc_all.barrier()

    return qc_all

#-----circuit-----
def get_common_form(x, qc_cosin, nqubit, qc_U=1, no_U=False):
    qr = QuantumRegister(nqubit, 'q')
    cr = ClassicalRegister(2, 'c')
    qc = QuantumCircuit(qr, cr)
    
    qc.initialize(x, range(nqubit))
    qc.barrier()
    
    # random unitary
    if not no_U:
        qc.compose(qc_U, inplace=True)
        qc.barrier()
    
    # nonlinear mapping unitary
    qc.compose(qc_cosin, inplace=True)

    return qc

def get_var_form(qc_train, nqubit, params):
    qc = qc_train.copy()
    # variational circuit
    qc_var = RealAmplitudes_circuit(nqubit, 'full', params)
    qc.compose(qc_var, inplace=True)

    qc.measure([0, 1], [0, 1])
    
    return qc

def get_class_from_counts(counts):
    z0 = {'0':0, '1':0}
    z1 = {'0':0, '1':0}
    counts_sum = sum(counts.values())
    for s in counts:
        z0[s[1]] += counts[s] / counts_sum
        z1[s[0]] += counts[s] / counts_sum

    z = [z0['0'] - z0['1'], z1['0'] - z1['1']]
    y_pre = np.exp(z) / np.sum(np.exp(z))

    return y_pre

def get_var_out(param):
    qc_train_U_i = param[0]
    nqubit = param[1]
    params = param[2]
    qc = get_var_form(qc_train_U_i, nqubit, params)
        
    #display(qc.draw())
    counts = execute(qc, backend, shots=1024).result().get_counts()
    y_pre = get_class_from_counts(counts)

    return y_pre

def VQELM(params):
    global N_train, y, nqubit, qc_train_U, time_exp5_train
    
    time_1 = timer()
    y_pre_all = []
    

    '''
    # -----multiprocessing-----
    params_all = []
    for i in range(N_train):
        params_all.append([qc_train_U[i], nqubit, params])
    cpu_counts = mp.cpu_count()
    if N_train < cpu_counts:
        cpu_counts = N_train
    with mp.Pool(cpu_counts) as pool:
        y_pre_all = pool.map(get_var_out, params_all)'''

    '''
    # -----batch qc, IBMQ-----
    batch = np.random.choice(N_train, 20)
    qc_batch = []
    for i in batch:
        qc = get_var_form(qc_train_U[i], nqubit, params)
        qc_batch.append(qc)

    job_manager = IBMQJobManager()
    job_set_multiple = job_manager.run(qc_batch, backend=backend, max_experiments_per_job=20)
    results = job_set_multiple.results()
    for i in range(len(batch)):
        counts = results.get_counts(i)
        y_pre = get_class_from_counts(counts)
        y_pre_all.append(y_pre)
    print(results.combine_results().time_taken)
    time_2 = timer()'''

    # -----batch optimzation-----
    batch = np.random.choice(N_train, 5)
    qc_batch = []
    for i in batch:
        qc = get_var_form(qc_train_U[i], nqubit, params)
        qc_batch.append(qc)
    results = execute(qc_batch, backend, shots=1024).result()
    counts_all = results.get_counts()
    for i in range(len(batch)):
        counts = counts_all[i]
        y_pre = get_class_from_counts(counts)
        y_pre_all.append(y_pre)
    time_2 = timer()
    time_exp5_train += results.time_taken
    
    cost = log_loss(y[batch], y_pre_all)
    print(time_2 - time_1, cost, ACC(y[batch], y_pre_all))

    return cost


def VQELM_noU(params):
    global N_train, y, nqubit, qc_train_noU, time_exp6_train

    time_1 = timer()
    y_pre_all = []

    '''
    params_all = []
    for i in range(N_train):
        params_all.append([qc_train_noU[i], nqubit, params])
    cpu_counts = mp.cpu_count()
    if N_train < cpu_counts:
        cpu_counts = N_train
    with mp.Pool(cpu_counts) as pool:
        y_pre_all = pool.map(get_var_out, params_all)'''

    # -----batch optimzation-----
    batch = np.random.choice(N_train, 5)
    qc_batch = []
    for i in batch:
        qc = get_var_form(qc_train_noU[i], nqubit, params)
        qc_batch.append(qc)
    results = execute(qc_batch, backend, shots=1024).result()
    counts_all = results.get_counts()
    for i in range(len(batch)):
        counts = counts_all[i]
        y_pre = get_class_from_counts(counts)
        y_pre_all.append(y_pre)
    time_2 = timer()
    time_exp6_train += results.time_taken
    
    cost = log_loss(y[batch], y_pre_all)
    print(time_2 - time_1, cost, ACC(y[batch], y_pre_all))

    return cost


if __name__ == '__main__':
    # ----------define----------
    pi = math.pi
    save_fig_flag = True
    save_excel_flag = True
    show_fig_flag = False
    path = 'figure_final/'

    '''
    dataset_names = ['haberman', 'transfusion', 'mammo', 'monk', 'forestfires', 'statlog', 'breastcarcer', 'breastcarcer_coimbra',\
                     'churn', 'electrical_grid', 'magic', 'thoraricsurgery', 'phishing', 'hcc', 'spambase', 'musk', 'madelon']'''

    dataset_names = ['thoraricsurgery', 'hcc', 'madelon']

    for dataset_name in dataset_names:
        if save_excel_flag:
            wb = Workbook()
            ws = wb.active
            ws.append(['ACC_1', 'ACC_2', 'ACC_5', 'ACC_6',\
                       'Ex_1_time_train', 'Ex_1_time_test', 'Ex_2_time_train', 'Ex_2_time_test',\
                       'Ex_5_time_train', 'Ex_5_time_test', 'Ex_6_time_train', 'Ex_6_time_test'])

        # ----------main process----------
        a = 20

        best_ACC_5 = 0
        best_ACC_6 = 0
        for row in range(1, 21):
            # ----------dataset----------
            #dataset_name = 'breastcarcer'

            if 'breastcarcer' in dataset_name:  # 489 210 16 1
                from VQELM_classification.figure_final.breastcarcer import fig
                x, x_test, y, y_test = dataset.load_breastcarcer_dataset()
            elif 'breastcarcer_coimbra' in dataset_name:  # 489 210 16 1
                from VQELM_classification.figure_final.breastcarcer_coimbra import fig
                x, x_test, y, y_test = dataset.load_breastcarcer_coimbra_dataset()
            elif 'churn' in dataset_name:  # 2205 945 16 1
                from VQELM_classification.figure_final.churn import fig
                x, x_test, y, y_test = dataset.load_churn_dataset()
            elif 'electrical_grid' in dataset_name:  # 7000 3000 16 1
                from classification.figure_final.electrical_grid import fig
                x, x_test, y, y_test = dataset.load_electrical_grid_dataset()
            elif 'forestfires' in dataset_name:  # 170 73 16 1
                from VQELM_classification.figure_final.forestfires import fig
                x, x_test, y, y_test = dataset.load_forestfires_dataset()
            elif 'haberman' in dataset_name:  # 214 92 4 1
                from VQELM_classification.figure_final.haberman import fig
                x, x_test, y, y_test = dataset.load_haberman_dataset()
            elif 'hcc' in dataset_name:  # 115 50 64 1
                from VQELM_classification.figure_final.hcc import fig
                x, x_test, y, y_test = dataset.load_hcc_dataset()
            elif 'madelon' in dataset_name:  # 1820 780 512 1
                from VQELM_classification.figure_final.madelon import fig
                x, x_test, y, y_test = dataset.load_madelon_dataset()
            elif 'magic' in dataset_name:  # 13314 5706 16 1
                from VQELM_classification.figure_final.magic import fig
                x, x_test, y, y_test = dataset.load_magic_dataset()
            elif 'mammo' in dataset_name:  # 672 289 8 1
                from VQELM_classification.figure_final.mammo import fig
                x, x_test, y, y_test = dataset.load_mammo_dataset()
            elif 'monk' in dataset_name:  # 1197 514 8 1
                from VQELM_classification.figure_final.monk import fig
                x, x_test, y, y_test = dataset.load_monk_dataset()
            elif 'musk' in dataset_name:  # 4618 1980 256 1
                from VQELM_classification.figure_final.musk import fig
                x, x_test, y, y_test = dataset.load_musk_dataset()
            elif 'phishing' in dataset_name:  # 7738 3317 32 1
                from VQELM_classification.figure_final.phishing import fig
                x, x_test, y, y_test = dataset.load_phishing_dataset()
            elif 'spambase' in dataset_name:  # 3220 1381 64 1
                from VQELM_classification.figure_final.spambase import fig
                x, x_test, y, y_test = dataset.load_spambase_dataset()
            elif 'statlog' in dataset_name:  # 483 207 16 1
                from VQELM_classification.figure_final.statlog import fig
                x, x_test, y, y_test = dataset.load_statlog_dataset()
            elif 'thoraricsurgery' in dataset_name:  # 329 141 32 1 nice
                from VQELM_classification.figure_final.thoraricsurgery import fig
                x, x_test, y, y_test = dataset.load_thoraricsurgery_dataset()
            elif 'transfusion' in dataset_name:  # 523 225 8 1
                from VQELM_classification.figure_final.transfusion import fig
                x, x_test, y, y_test = dataset.load_transfusion_dataset()


            N_train = len(x)
            N_test = len(x_test)
            m = len(x[0])
            print('\ndataset, N_train, N_test, m, row: ', dataset_name, N_train, N_test, m, row)
            nqubit = int(math.log(m, 2))
            random_theta = np.random.uniform(0, 2*pi, nqubit)
            qc_U = Random_unitary(random_theta, nqubit)

            optimizer = SPSA(maxiter=int((N_train/5))*5)  #NFT

            # ----------unitary matrix-nonlinear----------
            # ---train
            qc_cosin = []
            for x_i in x:
                qc_cosin.append(Unitary_nonlinear(x_i, nqubit, [20, 201]))
            qc_train_U = []
            qc_train_noU = []
            for i in range(N_train):
                qc_train_U.append(get_common_form(x[i], qc_cosin[i], nqubit, qc_U))
                qc_train_noU.append(get_common_form(x[i], qc_cosin[i], nqubit, no_U=True))

            # ---test
            qc_cosin_test = []
            for x_test_i in x_test:
                qc_cosin_test.append(Unitary_nonlinear(x_test_i, nqubit, [20, 201]))
            qc_test_U = []
            qc_test_noU = []
            for i in range(N_test):
                qc_test_U.append(get_common_form(x_test[i], qc_cosin_test[i], nqubit, qc_U))
                qc_test_noU.append(get_common_form(x_test[i], qc_cosin_test[i], nqubit, no_U=True))


            plt.close('all')
            plt.figure(figsize=(12, 8))


            # ----------ELM----------
            # -----ELM train-----
            time_begin = timer()
            P, A, beta = elm.ELM_train(x, y, 100, m)  # C, L
            time_end = timer()
            time_exp1_train = time_end - time_begin

            # -----elm test-----
            time_begin = timer()
            y_predict = elm.ELM_predict(x_test, A, beta)
            time_end = timer()
            time_exp1_test = time_end - time_begin

            ACC_1 = ACC(y_test, y_predict)
            print('ELM train time, test time, test ACC:', time_exp1_train, time_exp1_test, ACC_1)

            ax = plt.subplot(221)
            ax.plot([i for i in range(N_test)], np.argmax(y_test, axis=1), label='Real', color=colors[0], linewidth=3)
            ax.plot([i for i in range(N_test)], np.argmax(y_predict, axis=1), label='Predict', color=colors[6], linewidth=3)
            Plt_set(ax, 'Test sample', 'Output class', 'Ex_1')


            # ----------ELM+U_random simulate (H=XW, H\beta=Y)----------
            time_begin = timer()
            # -----random unitary-----
            U = Random_unitary_classical(random_theta, nqubit)

            # -----train-----
            H = Get_H(x, U)
            beta = Get_belta(H, y)
            time_end = timer()
            time_exp2_train = time_end - time_begin

            # -----test-----
            time_begin = timer()
            H_test = Get_H(x_test, U)
            Qelm_y_predict = H_test.dot(beta)
            time_end = timer()
            time_exp2_test = time_end - time_begin

            # -----result, figure-----
            ACC_2 = ACC(y_test, Qelm_y_predict)
            print('ELM+U_random train time, test time, test ACC:', time_exp2_train, time_exp2_test, ACC_2)

            ax = plt.subplot(222)
            plt.plot([i for i in range(N_test)], np.argmax(y_test, axis=1), label='Real', color=colors[0], linewidth=3)
            plt.plot([i for i in range(N_test)], np.argmax(Qelm_y_predict, axis=1), label='Predict', color=colors[6], linewidth=3)
            Plt_set(ax, 'Test sample', 'Output class', 'Ex_2')


            # ----------Quantum ELM+U_random simulate (H=XW, H\beta=Y)----------
            # -----train-----
            time_exp5_train = 0
            nparam = nqubit * 5 + nqubit
            init_params = np.random.rand(nparam)
            ret = optimizer.optimize(num_vars=nparam, objective_function=VQELM, initial_point=init_params)

            # -----test-----
            time_exp5_test = 0
            Qelm_y_predict = []
            for i in range(N_test):
                qc = get_var_form(qc_test_U[i], nqubit, ret[0])
                
                results = execute(qc, backend, shots=1024).result()
                counts = results.get_counts()
                y_pre = get_class_from_counts(counts)
                Qelm_y_predict.append(y_pre)

                time_exp5_test += results.time_taken

            Qelm_y_predict = np.array(Qelm_y_predict)

            # -----result, figure-----
            ACC_5 = ACC(y_test, Qelm_y_predict)
            print('QELM+U_random train time, test time, test ACC:', time_exp5_train, time_exp5_test, ACC_5)

            ax = plt.subplot(223)
            plt.plot([i for i in range(N_test)], np.argmax(y_test, axis=1), label='Real', color=colors[0], linewidth=3)
            plt.plot([i for i in range(N_test)], np.argmax(Qelm_y_predict, axis=1), label='Predict', color=colors[6], linewidth=3)
            Plt_set(ax, 'Test sample', 'Output class', 'Ex_5')


            # ----------Quantum ELM simulate (H=XW, H\beta=Y)----------
            # -----train-----
            time_exp6_train = 0
            nparam = nqubit * 5 + nqubit
            init_params = np.random.rand(nparam)
            ret = optimizer.optimize(num_vars=nparam, objective_function=VQELM_noU, initial_point=init_params)

            # -----test-----
            time_exp6_test = 0
            Qelm_y_predict = []
            for i in range(N_test):
                qc = get_var_form(qc_test_noU[i], nqubit, ret[0])
                
                results = execute(qc, backend, shots=1024).result()
                counts = results.get_counts()
                y_pre = get_class_from_counts(counts)
                Qelm_y_predict.append(y_pre)

                time_exp6_test += results.time_taken

            Qelm_y_predict = np.array(Qelm_y_predict)

            # -----result, figure-----
            ACC_6 = ACC(y_test, Qelm_y_predict)
            print('QELM train time, test time, test ACC:', time_exp6_train, time_exp6_test, ACC_6)

            ax = plt.subplot(224)
            plt.plot([i for i in range(N_test)], np.argmax(y_test, axis=1), label='Real', color=colors[0], linewidth=3)
            plt.plot([i for i in range(N_test)], np.argmax(Qelm_y_predict, axis=1), label='Predict', color=colors[6], linewidth=3)
            Plt_set(ax, 'Test sample', 'Output class', 'Ex_6')


            if ACC_5 > best_ACC_5 or ACC_6 > best_ACC_6:
                best_ACC_5 = ACC_5
                best_ACC_6 = ACC_6
                # ----------save-----------
                # -----fig------
                if save_fig_flag:
                    fig_path = path + dataset_name + '/fig/' + str(ACC_1) + '_' + str(ACC_2) + '_' + str(ACC_5) + '_' + str(ACC_6)
                    if row == 1:
                        del_file(path + dataset_name + '/fig')
                    plt.tight_layout()
                    plt.savefig(fig_path + '.pdf', dpi=600, bbox_inches='tight', pad_inches=0.05)

                if show_fig_flag:
                    plt.show()

            # -----excel-----
            if save_excel_flag:
                ws.append([ACC_1, ACC_2, ACC_5, ACC_6, time_exp1_train, time_exp1_test, \
                           time_exp2_train, time_exp2_test, time_exp5_train, time_exp5_test, time_exp6_train, time_exp6_test])
        if save_excel_flag:
            wb.save(path + dataset_name + '/results.xlsx')
            fig.Plot_data()
