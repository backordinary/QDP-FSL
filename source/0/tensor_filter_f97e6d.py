# https://github.com/BOBO1997/sigqs03/blob/b27cda80bc590382c951be175ccd26795e148e39/libmitigation/others/tensor_filter.py
from typing import Union, List
import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import pickle
from pprint import pprint
import time
import pdb
import heapq
from heapq import heappush, heappop

import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute, IBMQ
from qiskit.tools.visualization import plot_histogram
from qiskit.tools.monitor import job_monitor
import qiskit.providers.aer.noise as noise
import qiskit.ignis.mitigation as mit
from qiskit.ignis.mitigation.measurement import tensored_meas_cal, TensoredMeasFitter
import qiskit.circuit.library.standard_gates as gates

class priority_queue(object):
    """
    Priority queue wrapper which enables to compare the specific elements of container as keys.
    """
    def __init__(self, key_index = 0):
        """
        Arguments
            key_index: the index of elements as keys
        """
        self.key = lambda item: item[key_index]
        self.index = 0
        self.data = []

    def size(self):
        """
        Return the size of heap
        """
        return len(self.data)

    def push(self, item):
        """
        Push a container to heap list
        
        Arguments
            item: container
        """
        heapq.heappush(self.data, (self.key(item), self.index, item))
        self.index += 1

    def pop(self):
        """
        Pop the smallest element of heap
        """
        if len(self.data) > 0:
            return heapq.heappop(self.data)[2]
        else:
            return None
        
    def top(self):
        """
        Refer the smallest element of heap
        """
        if self.size() > 0:
            return self.data[0][2]
        else:
            return None

def draw_heatmap(data, row_labels = None, column_labels = None, norm=Normalize(vmin=-1, vmax=1)):
    
    if row_labels is None:
        row_labels = list(range(len(data)))
    if column_labels is None:
        column_labels = list(range(len(data[0])))
    # drawing part
    fig, ax = plt.subplots()
    # heatmap = ax.pcolor(data, cmap=plt.cm.Blues)
    heatmap = ax.pcolor(data, cmap="bwr", norm=norm)
    fig.colorbar(heatmap, ax=ax)

    ax.set_xticks(np.arange(data.shape[0]) + 0.5, minor=False)
    ax.set_yticks(np.arange(data.shape[1]) + 0.5, minor=False)

    ax.invert_yaxis()
    ax.xaxis.tick_top()

    ax.set_xticklabels(row_labels, minor=False)
    ax.set_yticklabels(column_labels, minor=False)
    plt.show()
    # plt.savefig('image.png')

    return heatmap


class TensoredMitigation():
    
    # OK
    def __init__(self, 
                 num_clbits: int, 
                 cal_matrices: List[np.array],
                 mit_pattern: List[List[int]] = None,
                 meas_layout: List[int] = None) -> None:
        """
        Initialize the TensoredMitigation class
        
        Arguments
            num_clbits: number of measured qubits (int)
            cal_matrices: calibration matrices (list of 2 * 2 numpy array)
            meas_layout: the mapping from classical registers to qubits
        """
        self.num_clbits = num_clbits
        self.cal_matrices = cal_matrices
        self.mit_pattern = mit_pattern if mit_pattern is not None else [[i] for i in range(self.num_clbits)]
        self.pinv_matrices = None
        self.svd_matrices = None
        self.Us = None
        self.Sigmas = None
        self.Vs = None
        self.pinv_svd_matrices = None
        self.pinvUs = None
        self.pinvSigmas = None
        self.pinvVs = None
        
        if meas_layout is None:
            meas_layout = [i for i in range(self.num_clbits)]
        meas_layout = meas_layout[::-1]  # reverse endian
        
        self.qubits_to_clbits = [-1 for _ in range(max(meas_layout) + 1)]
        for i, qubit in enumerate(meas_layout):
            self.qubits_to_clbits[qubit] = i
            
        # compute pseudo inverse and svd of all calibartion matrices
        self.run_inv_svd()

    # OK
    def kron_matrices(self, matrices: list):
        """
        Tensor product of given matrices (in the given order)
        """
        ret = matrices[0]
        for mat in matrices[1:]:
            ret = np.kron(ret, mat)
        return ret
    
    # OK
    def run_inv(self) -> None:
        """
        Prepare inverse of calibration matrices.
        
        Internal Method.
        O(n) time, O(n) space
        """
        if self.pinv_matrices is None:
            self.pinv_matrices = list(map(np.linalg.pinv, self.cal_matrices))
            
    # OK
    def run_svd(self) -> None:
        """
        Prepare svd of calibration matrices.
        
        Internal Method.
        O(n) time, O(n) space
        """
        if self.svd_matrices is None:
            self.svd_matrices = list(map(np.linalg.svd, self.cal_matrices))
            self.Us = [U for U , _, _ in self.svd_matrices]
            self.Sigmas = [np.diag(Sigma) for _, Sigma, _ in self.svd_matrices]
            self.Vs = [V.T for _ , _, V in self.svd_matrices]
            for i in range(len(self.mit_pattern)):
                negative_H = - np.array([[1,1],[1,-1]])
                if ( self.Us[i] == np.diag([1] * (2 ** len(self.mit_pattern[i]))) ).all() and ( self.Vs[i] == np.diag([1] * (2 ** len(self.mit_pattern[i]))) ).all():
                    print("changed identity matrix into negative Hadamard matrix at mit_pattern[", i, "]")
                    self.Us[i] = self.kron_matrices([negative_H] * len(self.mit_pattern[i])) / np.sqrt(2 ** len(self.mit_pattern[i]))
                    self.Vs[i] = self.kron_matrices([negative_H] * len(self.mit_pattern[i])) / np.sqrt(2 ** len(self.mit_pattern[i]))

    # OK
    def run_inv_svd(self) -> None:
        """
        Do singular value decomposition of all inverse calibration matrices.
        
        Internal Method.
        O(n) time, O(n) space
        """
        self.run_inv()
        self.run_svd()
        if self.pinv_svd_matrices is None:
            # self.pinv_svd_matrices = list(map(np.linalg.svd, self.pinv_matrices))
            # self.pinvUs = [U for U , _, _ in self.pinv_svd_matrices]
            # self.pinvSigmas = [np.diag(Sigma) for _, Sigma, _ in self.pinv_svd_matrices]
            # self.pinvVs = [V.T for _ , _, V in self.pinv_svd_matrices]
            self.pinvUs = list(map(np.linalg.pinv, self.Us))
            self.pinvSigmas = list(map(np.linalg.pinv, self.Sigmas))
            self.pinvVs = list(map(np.linalg.pinv, self.Vs))

    # OK
    def index_of_mat(self, state: Union[str, int], pos_clbits: List[int]) -> int:
        """
        Compute the index of (pseudo inverse) calibration matrix for the input quantum state
        If input state is string, then this process costs O(n) time, O(n) space
        If input state is integer, then this process costs O(len(pos_clbits)) time, O(1) space
        Using tensored mitigation, len(pos_clbits) == 1, therefore the time complexity would be O(1).
        
        Arguments
            state: the quantum states we focusing
            pos_clbits: the classical qubits of which the matrix is in charge (equivalent to indicating the matrix)
        Returns
            sub_state: the position in the matrix
        """
        if isinstance(state, str): # O(n) time, O(n) space
            sub_state = ""
            for pos in pos_clbits:
                sub_state += state[pos]
            return int(sub_state, 2)
        else:
            sub_state = 0
            for pos in pos_clbits: # O(len(pos_clbits)) time, O(1) space
                sub_state <<= 1
                sub_state += (state >> (self.num_clbits - pos - 1)) & 1
            return sub_state
    
    # OK
    def mitigate_one_state(self, target_state: Union[str, int], counts: dict) -> float:
        """
        Mitigate one state using inverse calibration matrices.
        O(n * shots) time, O(shots) space
        
        Arguments
            target_state: quanutum state to be mitigated (str or int)
            counts: raw counts (dict)
        Returns
            new_count: mitigated count of target state (float)
        """
        new_count = 0
        for source_state in counts: # O(shots)
            tensor_elem = 1.
            for pinv_mat, pos_qubits in zip(self.pinv_matrices, self.mit_pattern): # O(n) time
                pos_clbits = [self.qubits_to_clbits[qubit] for qubit in pos_qubits] # if completely tensored, then len(pos_clbits) == 1 -> O(1) time
                first_index = self.index_of_mat(target_state, pos_clbits) # O(1) time
                second_index = self.index_of_mat(source_state, pos_clbits) # O(1) time
                tensor_elem *= pinv_mat[first_index, second_index] # O(1) time
            new_count += tensor_elem * counts[source_state] # O(1) time
        return new_count

    # OK
    def col_basis(self, col_idx: int, labels: List[int], pinv_mats: List[np.array]) -> dict:
        """
        Coordinate transformation from pinv v basis to e basis
        O(s * n) time, O(s * n) space

        Arguments
            col_idx: 
            labels: 
        Returns
            col_i: a (col_idx)-th vector of pinvVs (default type is dict)
        """
        col_i = {label: 0 for label in labels}
        for label in labels:  # O(s) times
            tensor_elem = 1.
            for pinv_mat, pos_qubits in zip(pinv_mats, self.mit_pattern):  # O(n) time
                # if completely tensored, then len(pos_clbits) == 1 -> O(1) time
                pos_clbits = [self.qubits_to_clbits[qubit] for qubit in pos_qubits]
                first_index = self.index_of_mat(label, pos_clbits)  # O(1) time
                second_index = self.index_of_mat(col_idx, pos_clbits)  # O(1) time
                tensor_elem *= pinv_mat[first_index, second_index]  # O(1) time
            col_i[label] = tensor_elem

        return col_i

    # OK
    def v_basis(self, col_idx: int, labels: List[int]) -> dict:
        return self.col_basis(col_idx, labels, self.pinvVs)
    
    # OK
    def choose_vecs(self, state_idx: int, matrices: List[np.array]) -> List[np.array]:
        """
        O(n) time, O(n) space
        
        Arguments
            state_idx: the focusing state
            matrices: list of matrices to be choosed its row
        Returns
            vecs: the list of vector
        """
        vecs = []
        for mat, pos_qubits in zip(matrices, self.mit_pattern):
            pos_clbits = [self.qubits_to_clbits[qubit] for qubit in pos_qubits]
            vecs.append(mat[self.index_of_mat(state_idx, pos_qubits)])
        return vecs
    
    # OK
    def sum_of_tensored_vector(self, vecs: list) -> int:
        """
        O(n) time, O(n) space
        
        Arguments
            vecs: the vectors consisted of the tensor product
        Returns
            sum_val: the sum of tensored vector of input vectors
        """
        sum_val = 1
        for vec in reversed(vecs):
            sum_val *= sum(vec)
        return sum_val

    # OK
    def x_tilde_lm(self, y: dict) -> dict:
        """
        Use the priority queue to store the positive values with labels in raw counts y.
        O(shots * n * 2^n) time, O(s) space
        
        Arguments
            y: sparse probability vector (dict of str to int)
        Returns
            x_tilde: mitigated probability vector (dict of str to int)
        """
        print("Restriction to labels of y + Lagrange Multiplier + SGS algorithm")

        # preprocess 1
        # O(s * n * 2^n) time
        # compute sum of x
        sum_of_x = 0
        x = {state_idx: 0 for state_idx in range(2 ** self.num_clbits)}  # O(s) space # e basis
        for state_idx in range(2 ** self.num_clbits):  # O(s) time
            sum_of_col = self.sum_of_tensored_vector(self.choose_vecs(state_idx, self.pinv_matrices))  # O(n) time
            sum_of_x += sum_of_col * y[state_idx]
            x[state_idx] = self.mitigate_one_state(state_idx, y)  # O(n * s) time
        print("sum of mitigated probability vector x_s:", sum(x.values()))

        # exponential time computation
        # O(n * n * 2^n) time
        # compute the denominator of delta naively
        delta_denom = 0
        for state_idx in range(2 ** self.num_clbits):  # O(2^n)
            sum_of_vi = self.sum_of_tensored_vector(self.choose_vecs(state_idx, self.pinvVs))  # O(n) time
            lambda_i = self.sum_of_tensored_vector(self.choose_vecs(state_idx, self.pinvSigmas))  # O(n) time
            delta_denom += (sum_of_vi ** 2) / (lambda_i ** 2)
        delta_coeff = (1 - sum_of_x) / delta_denom  # O(1) time

        # exponential time computation
        # O(n * n * 2^n) time
        # prepare x_hat_s naively
        x_hat = copy.deepcopy(x)
        for col_idx in range(2 ** self.num_clbits):  # O(2^n)
            sum_of_vi = self.sum_of_tensored_vector(self.choose_vecs(col_idx, self.pinvVs))  # O(n) time
            lambda_i = self.sum_of_tensored_vector(self.choose_vecs(col_idx, self.pinvSigmas))  # O(n) time
            delta_col = sum_of_vi / (lambda_i ** 2)
            v_col = self.v_basis(0, x_hat.keys())
            for state_idx in x_hat:
                x_hat[state_idx] += delta_coeff * delta_col * v_col[state_idx]

        # algorithm by Smolin et al.
        # print(x_hat)
        return self.sgs_algorithm(x_hat)

    # OK
    def x_tilde_s_lm(self, y: dict) -> dict:
        """
        Use the priority queue to store the positive values with labels in raw counts y.
        O(shots * n * 2^n) time, O(s) space
        
        Arguments
            y: sparse probability vector (dict of str to int)
        Returns
            x_tilde: mitigated probability vector (dict of str to int)
        """
        print("Restriction to labels of y + Lagrange Multiplier + SGS algorithm")
        
        # preprocess 1
        # O(s * s * n) time
        # compute sum of x
        sum_of_x = 0
        x_s = {state_idx: 0 for state_idx in y} # O(s) space # e basis
        for state_idx in y: # O(s) time
            sum_of_col = self.sum_of_tensored_vector(self.choose_vecs(state_idx, self.pinv_matrices)) # O(n) time
            sum_of_x += sum_of_col * y[state_idx]
            x_s[state_idx] = self.mitigate_one_state(state_idx, y) # O(n * s) time
        print("sum of mitigated probability vector x_s:", sum(x_s.values()))
        
        # exponential time computation
        # O(n * n * 2^n) time
        # compute the denominator of delta naively
        delta_denom = 0
        for state_idx in range(2 ** self.num_clbits): #  O(2^n)
            sum_of_vi = self.sum_of_tensored_vector(self.choose_vecs(state_idx, self.pinvVs)) # O(n) time
            lambda_i = self.sum_of_tensored_vector(self.choose_vecs(state_idx, self.pinvSigmas)) # O(n) time
            delta_denom += (sum_of_vi ** 2) / (lambda_i ** 2)
        delta_coeff = (1 - sum_of_x) / delta_denom # O(1) time
        
        # exponential time computation
        # O(n * n * 2^n) time
        # prepare x_hat_s naively
        x_hat_s = copy.deepcopy(x_s)
        for col_idx in range(2 ** self.num_clbits): # O(2^n)
            sum_of_vi = self.sum_of_tensored_vector(self.choose_vecs(col_idx, self.pinvVs)) # O(n) time
            lambda_i = self.sum_of_tensored_vector(self.choose_vecs(col_idx, self.pinvSigmas)) # O(n) time
            delta_col = sum_of_vi / (lambda_i ** 2)
            v_col = self.v_basis(0, x_hat_s.keys())
            for state_idx in x_hat_s:
                x_hat_s[state_idx] += delta_coeff * delta_col * v_col[state_idx]

        # algorithm by Smolin et al.
        print(x_hat_s)
        return self.sgs_algorithm(x_hat_s)
        
    # OK
    def x_tilde_s_lm_s(self, y: dict) -> dict:
        """
        Use the priority queue to store the positive values with labels in raw counts y.
        O() time, O(shots) space
        
        Arguments
            y: sparse probability vector (dict of str to int)
        Returns
            x_tilde: mitigated probability vector (dict of str to int)
        """
        print("Restriction to labels of y + efficient Lagrange Multiplier + SGS algorithm")
        
        t1 = time.time()

        # preprocess 1
        # O(s * s * n) time
        # compute sum of x
        sum_of_x = 0
        x_s = {state_idx: 0 for state_idx in y} # O(s) space # e basis
        for state_idx in y: # O(s) time
            sum_of_col = self.sum_of_tensored_vector(self.choose_vecs(state_idx, self.pinv_matrices)) # O(n) time
            sum_of_x += sum_of_col * y[state_idx] # O(1) time
            x_s[state_idx] = self.mitigate_one_state(state_idx, y) # O(n * s) time
        
        t2 = time.time()
        print("sum of mitigated probability vector x_s:", sum(x_s.values()))
        print("t2 - t1:", t2 - t1)

        # compute the denominator of delta efficiently
        # O(n) time
        sum_of_vi = self.sum_of_tensored_vector(self.choose_vecs(0, self.pinvVs)) # O(n) time
        lambda_i = self.sum_of_tensored_vector(self.choose_vecs(0, self.pinvSigmas)) # O(n) time
        delta_denom = (sum_of_vi ** 2) / (lambda_i ** 2) # O(1) time
        delta_coeff = (1 - sum_of_x) / delta_denom # O(1) time
        
        t3 = time.time()
        print("t3 - t2:", t3 - t2)

        # prepare x_hat_s efficiently # only choose column 0
        # did not transform the coordinate.
        # O(s * n) time
        x_hat_s = copy.deepcopy(x_s)
        sum_of_vi = self.sum_of_tensored_vector(self.choose_vecs(0, self.pinvVs)) # O(n) time
        lambda_i = self.sum_of_tensored_vector(self.choose_vecs(0, self.pinvSigmas)) # O(n) time
        delta_col = sum_of_vi / (lambda_i ** 2) # O(1) time
        v_col = self.v_basis(0, x_hat_s.keys()) # O(s) time
        for state_idx in x_hat_s: # O(s) time
            x_hat_s[state_idx] += delta_coeff * delta_col * v_col[state_idx] # O(1) time

        t4 = time.time()
        print("sum of mitigated probability vector x_hat_s:", sum(x_hat_s.values()))
        print("t4 - t3:", t4 - t3)

        # algorithm by Smolin et al. # O(s * log s) time
        return self.sgs_algorithm(x_hat_s)

    # OK
    def x_tilde_s(self, y: dict) -> dict:
        """
        Remove the negative elements from mitigated probability vector with sum(x_sorted) = 1
        x_sorted: sorted probability vector with sum(x_sorted) = 1, x_sorted is sorted in the descending order
        
        Arguments
            y: sparse probability vector (dict of str to int)
            shots: 
        Returns
            x_tilde: mitigated probability vector (dict of str to int)
        """
        print("Restriction to labels of y + SGS algorithm")
        # O(s * s * n) time
        x_s = {state_idx: 0 for state_idx in y} # O(s) space # e basis
        for state_idx in y: # O(s) time
            x_s[state_idx] = self.mitigate_one_state(state_idx, y) # O(n * s) time
        print("sum of mitigated probability vector x_s:", sum(x_s.values()))

        # algorithm by Smolin et al. # O(n * 2^n) time
        return self.sgs_algorithm(x_s)
    
    def x_tilde_k0(self, y: dict) -> dict:
        """
        Arguments
            y: sparse probability vector (dict of str to int)
            shots: 
        Returns
            x_tilde: mitigated probability vector (dict of str to int)
        """
        print("Low rank approximation + Lagrange Multiplier + SGS algorithm")
        
        # preprocess 1
        # O(2^n) time
        # compute sum of x
        x = {state_idx: 0 for state_idx in range(2 ** self.num_clbits)} # O(2^n) space # e basis

        inv_sigma_0 = 1
        for smat in self.pinvSigmas:
            inv_sigma_0 *= smat[0,0]
        inv_sigma_0 = 1 / inv_sigma_0
        u_0T = self.col_basis(0, range(2 ** self.num_clbits), self.pinvUs)
        u_0Ty = 0
        for u_val, y_val in zip(u_0T.values(), y.values()):
            u_0Ty += u_val * y_val
        v_0 = self.v_basis(0, x.keys())
        coeff = u_0Ty * inv_sigma_0
        for state_idx in range(self.num_clbits): # O(s) time
            x[state_idx] = coeff * v_0[state_idx]
        sum_of_x = sum(x.values())
        print("sum of mitigated probability vector x:", sum_of_x)
        
        # exponential time computation
        # O(n * 2^n) time
        # compute Delta_0 and correct x
        delta_denom = 0
        sum_of_v0 = self.sum_of_tensored_vector(self.choose_vecs(0, self.pinvVs)) # O(n) time
        delta_0 = (1 - sum_of_x) / (sum_of_v0 ** 2) # O(1) time
        for state_idx in x:
            x[state_idx] += delta_0 * v_0[state_idx]
        print("sum of mitigated probability vector x_hat:", sum_of_x)

        # algorithm by Smolin et al.
        # print(x_hat)
        return self.sgs_algorithm(x)

    def x_tilde_k(self, y: dict, k=1) -> dict:
        """
        Arguments
            y: sparse probability vector (dict of str to int)
            shots: 
        Returns
            x_tilde: mitigated probability vector (dict of str to int)
        """
        print("Low rank approximation + Lagrange Multiplier + SGS algorithm")

        # preprocess 1
        # O(2^n) time
        # compute sum of x
        x = {state_idx: 0 for state_idx in range(2 ** self.num_clbits)}  # O(2^n) space # e basis

        inv_sigma_0 = 1
        for smat in self.pinvSigmas:
            inv_sigma_0 *= smat[0, 0]
        inv_sigma_0 = 1 / inv_sigma_0
        u_0T = self.col_basis(0, range(2 ** self.num_clbits), self.pinvUs)
        u_0Ty = 0
        for u_val, y_val in zip(u_0T.values(), y.values()):
            u_0Ty += u_val * y_val
        v_0 = self.v_basis(0, x.keys())
        coeff = u_0Ty * inv_sigma_0
        for state_idx in range(self.num_clbits):  # O(s) time
            x[state_idx] = coeff * v_0[state_idx]
        sum_of_x = sum(x.values())
        print("sum of mitigated probability vector x:", sum_of_x)

        # exponential time computation
        # O(n * 2^n) time
        # compute Delta_0 and correct x
        delta_denom = 0
        sum_of_v0 = self.sum_of_tensored_vector(
            self.choose_vecs(0, self.pinvVs))  # O(n) time
        delta_0 = (1 - sum_of_x) / (sum_of_v0 ** 2)  # O(1) time
        for state_idx in x:
            x[state_idx] += delta_0 * v_0[state_idx]
        print("sum of mitigated probability vector x_hat:", sum_of_x)

        # algorithm by Smolin et al.
        # print(x_hat)
        return self.sgs_algorithm(x)

    def x_tilde_k_s(self, y: dict) -> dict:
        """
        Arguments
            y: sparse probability vector (dict of str to int)
            shots: 
        Returns
            x_tilde: mitigated probability vector (dict of str to int)
        """
        print("Low rank approximation + Restriction to labels of y + Lagrange Multiplier + SGS algorithm")
        # preprocess 1
        # O(s * s * n) time
        # compute sum of x
        sum_of_x = 0
        x_s = {state_idx: 0 for state_idx in y} # O(s) space # e basis
        for state_idx in y: # O(s) time
            sum_of_col = self.sum_of_tensored_vector(self.choose_vecs(state_idx, self.pinv_matrices)) # O(n) time
            sum_of_x += sum_of_col * y[state_idx]
            x_s[state_idx] = self.mitigate_one_state(state_idx, y) # O(n * s) time
        print("sum of mitigated probability vector x_s:", sum(x_s.values()))
        
        # exponential time computation
        # O(n * n * 2^n) time
        # compute the denominator of delta naively
        delta_denom = 0
        for state_idx in range(2 ** self.num_clbits): #  O(2^n)
            sum_of_vi = self.sum_of_tensored_vector(self.choose_vecs(state_idx, self.pinvVs)) # O(n) time
            lambda_i = self.sum_of_tensored_vector(self.choose_vecs(state_idx, self.pinvSigmas)) # O(n) time
            delta_denom += (sum_of_vi ** 2) / (lambda_i ** 2)
        delta_coeff = (1 - sum_of_x) / delta_denom # O(1) time
        
        # exponential time computation
        # O(n * n * 2^n) time
        # prepare x_hat_s naively
        x_hat_s = copy.deepcopy(x_s)
        for col_idx in range(2 ** self.num_clbits): # O(2^n)
            sum_of_vi = self.sum_of_tensored_vector(self.choose_vecs(col_idx, self.pinvVs)) # O(n) time
            lambda_i = self.sum_of_tensored_vector(self.choose_vecs(col_idx, self.pinvSigmas)) # O(n) time
            delta_col = sum_of_vi / (lambda_i ** 2)
            v_col = self.v_basis(0, x_hat_s.keys())
            for state_idx in x_hat_s:
                x_hat_s[state_idx] += delta_coeff * delta_col * v_col[state_idx]

        # algorithm by Smolin et al.
        print(x_hat_s)
        return self.sgs_algorithm(x_hat_s)

    # OK
    def default(self, y: dict) -> dict:
        """
        Remove the negative elements from mitigated probability vector with sum(x_sorted) = 1
        x_sorted: sorted probability vector with sum(x_sorted) = 1, x_sorted is sorted in the descending order
        
        Arguments
            y: sparse probability vector (dict of str to int)
            shots: 
        Returns
            x_tilde: mitigated probability vector (dict of str to int)
        """
        print("x + SGS algorithm")
        # mitigate raw counts y using tensored mitigation
        # O(s * n * 2^n)
        x = {state_idx: 0 for state_idx in range(2 ** self.num_clbits)} # O(shots) space # e basis
        for state_idx in range(2 ** self.num_clbits): # O(2^n) time
            x[state_idx] = self.mitigate_one_state(state_idx, y) # O(s * n)
        print("sum of mitigated probability vector x:", sum(x.values()))

        # algorithm by Smolin et al. # O(n * 2^n) time
        return self.sgs_algorithm(x)
    
    # OK
    def sgs_algorithm(self, x_hat_s: dict) -> dict:
        """
        The negative cancellation algorithm by Smolin, Gambetta, and Smith.
        Arguments
            x_hat_s: sum 1 probability vecotor with negative values
        Returns
            x_tilde: physically correct probability vector
        """
        # preporcess 2
        # compute the number and the sum of negative values
        # O(shots) space
        pq = priority_queue(key_index=1)
        sum_of_x_hat = 0
        for state_idx in x_hat_s: # O(shots) time
            if x_hat_s[state_idx] > 0:
                pq.push( (state_idx, x_hat_s[state_idx]) ) # O(log(shots))
                sum_of_x_hat += x_hat_s[state_idx]
        print("number of positive values: ", pq.size())

        negative_accumulator = 1 - sum_of_x_hat 
        if negative_accumulator >= 0:
            print("accumulator is positive, we might even ignoring the necessal positive values.")
        
        while pq.size() > 0: # O(shots) time
            _, x_hat_i = pq.top()
            if x_hat_i + negative_accumulator / pq.size() < 0:
                negative_accumulator += x_hat_i
                _, _ = pq.pop() # O(log(shots))
                continue
            else:
                break
                
        x_tilde = {}
        denominator = pq.size()
        while pq.size() > 0: # O(shots) time
            state_idx, x_hat_i = pq.pop() # O(log(shots))
            x_tilde[state_idx] = x_hat_i + negative_accumulator / denominator
            
        return x_tilde
    
    # OK
    def apply(self,
              counts: dict,
              shots: int = None,
              method: str = "default") -> dict:
        """
        Do whole process of mitigation

        Arguments
            counts: raw counts (dict of str to int)
            shots: total number of shot (int)
            method: mitigation option (str)
            
                * "default": O(n * 2^n) time and O(2^n) space

                * "x_tilde_s": O(s * s * n) time and O(s) space
                
                * "x_tilde_lm": priority queue for labels in raw counts, O(n * s * 2^n) times and O(s) space
                
                * "x_tilde_s_lm": priority queue for labels in raw counts, O(s * s * n) times and O(s) space

                * "ss": reduce A into the elements indexed of ylabels, O(s * s * n) times and O(s * s) space
                
        Returns
            mitigated_counts: mitigated counts (dict of str to float)
        """
        
        if shots is None:
            shots = sum(counts.values())
        
        # make probability vector (dict)
        y = {int(state, 2) : counts[state] / shots for state in counts}

        new_counts = None
        if method == "x_tilde_lm": # exp time, naive lagrangian multiplier
            x_tilde = self.x_tilde_lm(y)
        elif method == "x_tilde_s_lm": # poly time, lagrangian multiplier in S
            x_tilde = self.x_tilde_s_lm(y)
        elif method == "x_tilde_s_lm_s": # poly time, lagrangian multiplier in S and only choose col 0
            x_tilde = self.x_tilde_s_lm_s(y)
        elif method == "x_tilde_s": # poly time, only use S
            x_tilde = self.x_tilde_s(y)
        elif method == "x_tilde_k": # ---
            x_tilde = self.x_tilde_k(y)
        elif method == "x_tilde_k_s": # ---
            x_tilde = self.x_tilde_k_s(y)
        elif method == "default": # exp time, only SGS algorithm
            x_tilde = self.default(y)
        else:
            print("NO SUCH METHOD")
        
        print("main process: Done!")
        mitigated_counts = {format(state, "0"+str(self.num_clbits)+"b") : x_tilde[state] * shots for state in x_tilde} # rescale to counts
        return mitigated_counts

    # ====================================== for debug ========================================== #
    # OK
    def listup_deltas(self, counts: dict, shots: int = None, basis: str = "v") -> List[float]:
        """
        List up all the Delta_i values to see their distribution
        For debug and inspection.
        """
        if shots is None:
            shots = sum(counts.values())
        y = {int(state, 2) : counts[state] / shots for state in counts}
        self.run_inv_svd()
        
        sum_of_x = 0
        delta_denom = 0
        deltas = [0 for _ in range(2 ** self.num_clbits)] # v basis
        for state_idx in range(2 ** self.num_clbits): # O(2^n) time
            sum_of_x += self.mitigate_one_state(state_idx, y) # O(n * shots) time
            sum_of_vi = self.sum_of_tensored_vector(self.choose_vecs(state_idx, self.pinvVs)) # O(n) time
            lambda_i = self.sum_of_tensored_vector(self.choose_vecs(state_idx, self.pinvSigmas)) # O(n) time
            delta_denom += (sum_of_vi ** 2) / (lambda_i ** 2) # O(1) time
            deltas[state_idx] = sum_of_vi / (lambda_i ** 2)
        delta_coeff = (1 - sum_of_x) / delta_denom # O(1) time
        deltas = [delta_coeff * delta_i for delta_i in deltas]
        print("sum_of_x: ", sum_of_x)
        print("delta_denom: ", delta_denom)
        print("delta_coeff: ", delta_coeff)
        
        self.inspect_sum_of_x(counts)
        
        if basis == "v":
            return deltas
        else:
            e_deltas = np.zeros(2 ** self.num_clbits) # O(2^n) space
            V = self.kron_matrices(self.pinvVs) # O(4^n) time, O(4^n) space
            _ = draw_heatmap(V, list(range(2 ** self.num_clbits)), list(range(2 ** self.num_clbits)))
            for i, vi in enumerate(V.T):
                e_deltas += deltas[i] * vi
            return e_deltas.tolist()
        
    # OK
    def listup_sum_of_cols(self, counts: dict, shots: int = None, method="naive") -> tuple:
        """
        List up all the Delta_i values to see their distribution
        For debug and inspection.
        """
        if shots is None:
            shots = sum(counts.values())
        y = {int(state, 2) : counts[state] / shots for state in counts}
        self.run_inv_svd()
        
        sum_of_pinv_vis = [self.sum_of_tensored_vector(self.choose_vecs(state_idx, self.pinvVs)) for state_idx in range(2 ** self.num_clbits)]
        pinv_lambdas = [self.sum_of_tensored_vector(self.choose_vecs(state_idx, self.pinvSigmas)) for state_idx in range(2 ** self.num_clbits)]
        
        return sum_of_pinv_vis, pinv_lambdas

    # OK
    def inspect_sum_of_x(self, counts: dict, shots: int = None) -> None:
        """
        For debug and inspection.
        """
        if shots is None:
            shots = sum(counts.values())
        y = {int(state, 2) : counts[state] / shots for state in counts}
        self.run_inv_svd()
        
        sum_of_x = 0
        for state_idx in range(2 ** self.num_clbits): # O(2^n) time
            sum_of_x += self.mitigate_one_state(state_idx, y)  # O(n * shots) time
        
        sum_x = 0
        for state_idx, value in y.items(): # O(shots) time
            sum_of_col = self.sum_of_tensored_vector(self.choose_vecs(state_idx, self.pinv_matrices)) # O(n) time
            sum_x += sum_of_col * value
            
        print("sum_of_x == sum_x")
        print("sum_of_x: ", sum_of_x)
        print("sum_x: ", sum_x)
        print(sum_of_x == sum_x)
