# https://github.com/wxzsan/ITE_proj/blob/7cf24c5035b187e6cebd9490f28116cd3ba6d9ee/vqa/gradient.py
import matplotlib.pyplot as plt
import numpy as np
from qiskit import *
from qiskit import Aer
from .ansatz import QuantumCircuitUnitary
import scipy.sparse as sparse
import scipy.sparse.linalg as sparse_alg
from scipy.sparse import csgraph



backend = Aer.get_backend('statevector_simulator')

#_qiskit suffix means that is is implemented by simulator in qiksit, _sparse suffix means that it is implemented by sparse matrix multiplication in numpy

def forward_qiskit(qc: QuantumCircuit, values_dict: dict) -> np.ndarray:

    qc_assigned = qc.assign_parameters(values_dict)
    job = execute(qc_assigned, backend)
    result = job.result()
    return result.get_statevector(qc_assigned)

def forward(qc: QuantumCircuitUnitary, value_list: list) -> np.ndarray:
    dim = 2**qc.qcnt
    state = np.zeros(dim)
    state[dim-1] = 1.
    gate_list = qc.gate_list
    num_gates = len(gate_list)
    k_params = 0
    for i in range(num_gates):
        if gate_list[i][1]:
            state = (np.cos(value_list[k_params]/2)*np.identity(dim) - 1j*np.sin(value_list[k_params]/2)*gate_list[i][0])@state
            k_params += 1
        else:
            state = gate_list[i][0]@state
    return state

def forward_sparse(qc: QuantumCircuitUnitary, value_list: list) -> np.ndarray:
    dim = 2**qc.qcnt
    state = np.zeros(dim)
    state[dim-1] = 1.
    gate_list = qc.gate_list
    num_gates = len(gate_list)
    k_params = 0
    for i in range(num_gates):
        if gate_list[i][1]:
            state = (np.cos(value_list[k_params]/2)*sparse.eye(dim) - 1j*np.sin(value_list[k_params]/2)*gate_list[i][0])@state
            k_params += 1
        else:
            state = gate_list[i][0]@state
    return state



def get_qfi_qiskit(qc, values_dict):
    # first calc par phi/par theta
    eps = 1e-6
    num_params = len(values_dict)
    current_state = forward(qc, values_dict)

    partial_dev = []
    for param in values_dict.keys():
        values_dict[param] += eps
        partial_dev.append((forward(qc, values_dict)-current_state)/eps)
        values_dict[param] -= eps

    qfi = np.zeros((num_params, num_params))

    for i in range(num_params):
        for j in range(num_params):
            qfi[i][j] = 4*np.real(partial_dev[i]@partial_dev[j].conj()-partial_dev[i]@current_state.conj()*current_state@partial_dev[j].conj())
    
    return qfi



def get_qfi(qc, value_list):
    # first calc par phi/par theta
    eps = 1e-6
    num_params = len(value_list)
    current_state = forward(qc, value_list)

    partial_dev = []
    for i in range(num_params):
        tmp = value_list[i]
        value_list[i] += eps
        partial_dev.append((forward(qc, value_list)-current_state)/eps)
        value_list[i] = tmp

    qfi = np.zeros((num_params, num_params))

    for i in range(num_params):
        for j in range(num_params):
            qfi[i][j] = 4*np.real(partial_dev[i]@partial_dev[j].conj()-partial_dev[i]@current_state.conj()*current_state@partial_dev[j].conj())
    
    return qfi

def get_qfi_sparse(qc, value_list):
    # first calc par phi/par theta
    eps = 1e-6
    num_params = len(value_list)
    current_state = forward_sparse(qc, value_list)

    partial_dev = []
    for i in range(num_params):
        tmp = value_list[i]
        value_list[i] += eps
        partial_dev.append((forward_sparse(qc, value_list)-current_state)/eps)
        value_list[i] = tmp

    qfi = np.zeros((num_params, num_params))

    for i in range(num_params):
        for j in range(num_params):
            qfi[i][j] = 4*np.real(partial_dev[i]@partial_dev[j].conj()-partial_dev[i]@current_state.conj()*current_state@partial_dev[j].conj())
    
    return qfi

