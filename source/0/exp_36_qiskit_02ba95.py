# https://github.com/weiwenjiang/qc_mnist/blob/c337d72c207fad7b998528e77795cbf0bf1f8fba/Revision/Revision_Exp2-QF-FB/exp_36_qiskit.py

import torch
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
import torch.nn as nn
import shutil
import os
import time
import sys
from pathlib import Path
import functools
print = functools.partial(print, flush=True)
import numpy as np
import argparse

from qiskit_library import *
import torch
import numpy as np
from random import randrange
import qiskit as qk
from qiskit import Aer
from qiskit import execute
import math
import sys
import random
import time
from tqdm import tqdm

from qiskit import IBMQ
# IBMQ.delete_accounts()
# IBMQ.save_account('62d0e14364f490e45b5b5e0f6eebdbc083270ffffb660c7054219b15c7ce99ab4aa3b321309c0a9d0c3fc20086baece1376297dcdb67c7b715f9de1e4fa79efb')
# IBMQ.load_account()


from lib_qc import *
from lib_util import *
from lib_net import *


import qiskit_library

from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister
from qiskit.extensions import XGate, UnitaryGate
import qiskit
from math import sqrt
import math
import copy


def get_index_list(input, target):
    index_list = []
    try:
        beg_pos = 0
        while True:
            find_pos = input.index(target, beg_pos)
            index_list.append(find_pos)
            beg_pos = find_pos + 1
    except Exception as exception:
        pass
    return index_list


def change_sign(sign, bin):
    affect_num = [bin]
    one_positions = []

    try:
        beg_pos = 0
        while True:
            find_pos = bin.index("1", beg_pos)
            one_positions.append(find_pos)
            beg_pos = find_pos + 1
    except Exception as exception:
        # print("Not Found")
        pass

    for k, v in sign.items():
        change = True
        for pos in one_positions:
            if k[pos] == "0":
                change = False
                break
        if change:
            sign[k] = -1 * v


def find_start(affect_count_table, target_num):
    for k in list(affect_count_table.keys())[::-1]:
        if target_num <= k:
            return k


def recursive_change(direction, start_point, target_num, sign, affect_count_table, quantum_gates):
    if start_point == target_num:
        # print("recursive_change: STOP")
        return

    gap = int(math.fabs(start_point - target_num))
    step = find_start(affect_count_table, gap)
    change_sign(sign, affect_count_table[step])
    quantum_gates.append(affect_count_table[step])

    if direction == "r":
        # print("recursive_change: From",start_point,"Right(-):",step)
        start_point = start_point - step
        direction = "l"
        recursive_change(direction, start_point, target_num, sign, affect_count_table, quantum_gates)

    else:
        # print("recursive_change: From",start_point,"Left(+):",step)
        start_point = start_point + step
        direction = "r"
        recursive_change(direction, start_point, target_num, sign, affect_count_table, quantum_gates)


def guarntee_upper_bound_algorithm(sign, target_num, total_len, digits):
    flag = "0" + str(digits) + "b"
    pre_num = 0
    affect_count_table = {}
    quantum_gates = []
    for i in range(digits):
        cur_num = pre_num + pow(2, i)
        pre_num = cur_num
        binstr_cur_num = format(cur_num, flag)
        affect_count_table[int(pow(2, binstr_cur_num.count("0")))] = binstr_cur_num

    if target_num in affect_count_table.keys():
        quantum_gates.append(affect_count_table[target_num])
        change_sign(sign, affect_count_table[target_num])
    else:
        direction = "r"
        start_point = find_start(affect_count_table, target_num)
        quantum_gates.append(affect_count_table[start_point])
        change_sign(sign, affect_count_table[start_point])
        recursive_change(direction, start_point, target_num, sign, affect_count_table, quantum_gates)

    return quantum_gates


def qf_map_extract_from_weight(weights):
    # Find Z control gates according to weights
    w = (weights.detach().cpu().numpy())
    total_len = len(w)
    target_num = np.count_nonzero(w == -1)
    if target_num > total_len / 2:
        w = w * -1
    target_num = np.count_nonzero(w == -1)
    digits = int(math.log(total_len, 2))
    flag = "0" + str(digits) + "b"
    max_num = int(math.pow(2, digits))
    sign = {}
    for i in range(max_num):
        sign[format(i, flag)] = +1
    quantum_gates = guarntee_upper_bound_algorithm(sign, target_num, total_len, digits)

    # for k,v in sign.items():
    #     print(k,v)
    # print(w)

    # Build the mapping from weight to final negative num
    fin_sign = list(sign.values())
    fin_weig = [int(x) for x in list(w)]

    # print(fin_sign)
    # print(fin_weig)
    sign_neg_index = []
    try:
        beg_pos = 0
        while True:
            find_pos = fin_sign.index(-1, beg_pos)
            # qiskit_position = int(format(find_pos,flag)[::-1],2)
            sign_neg_index.append(find_pos)
            beg_pos = find_pos + 1
    except Exception as exception:
        pass

    weight_neg_index = []
    try:
        beg_pos = 0
        while True:
            find_pos = fin_weig.index(-1, beg_pos)
            weight_neg_index.append(find_pos)
            beg_pos = find_pos + 1
    except Exception as exception:
        pass
    map = {}
    for i in range(len(sign_neg_index)):
        map[sign_neg_index[i]] = weight_neg_index[i]

    # print(map)

    ret_index = list([-1 for i in range(len(fin_weig))])

    for k, v in map.items():
        ret_index[k] = v

    for i in range(len(fin_weig)):
        if ret_index[i] != -1:
            continue
        for j in range(len(fin_weig)):
            if j not in ret_index:
                ret_index[i] = j
                break
    #
    # ret_index = list(range(len(fin_weig)))
    #
    # for k, v in map.items():
    #     tmp1 = ret_index[k]
    #     tmp2 = ret_index[v]
    #     ret_index[k] = tmp2
    #     ret_index[v] = tmp1



    return quantum_gates, ret_index


def extract_model(model):
    layer_prop = {}
    batch_adj_prop = {}
    indiv_adj_prop = {}
    for name, para in model.named_parameters():
        if "fc" in name:
            layer_id = int(name.split(".")[0].split("c")[1])
            layer_prop[layer_id] = [para.shape[1], para.shape[0], binarize(para)]
        elif "qca" in name:
            if "l_0_5" in name or "running_rot" in name:
                layer_id = int(name.split(".")[0].split("a")[1])
                layer_fun = name.split(".")[1]
                if layer_id not in batch_adj_prop.keys():
                    batch_adj_prop[layer_id] = {}
                batch_adj_prop[layer_id][layer_fun] = para
        else:
            print(name, para)

    # print(layer_prop)
    # print(batch_adj_prop)

    # First layer
    first_layer_num = layer_prop[0][1]

    first_layer_input_q = int(math.log(layer_prop[0][0], 2))
    first_layer_addition_q = max(first_layer_input_q - 2, 0)
    first_layer_batch_q = 0
    if 0 in batch_adj_prop.keys():
        first_layer_batch_q = 2

    first_layer_q = first_layer_num * (first_layer_input_q + first_layer_batch_q)

    # print(first_layer_q)

    # Second layer
    second_layer_num = layer_prop[1][1]
    if layer_prop[1][0] == 0:
        second_layer_input_q = int(math.log(layer_prop[1][0], 2))
        second_layer_encode_q = 0
        second_layer_output_q = 0
    else:
        second_layer_input_q = layer_prop[1][0]
        second_layer_encode_q = int(math.log(second_layer_input_q, 2))
        second_layer_output_q = int(math.log(second_layer_encode_q, 2))

    second_layer_addition_q = max(first_layer_input_q - 2, 0)

    second_layer_batch_q = 0
    if 1 in batch_adj_prop.keys():
        second_layer_batch_q = 2

    second_layer_q = second_layer_num * (second_layer_input_q + second_layer_output_q + second_layer_batch_q)
    # print(second_layer_q)
    sec_list = [second_layer_input_q, second_layer_num * second_layer_output_q,
                second_layer_num * second_layer_encode_q, second_layer_num * second_layer_batch_q]
    return first_layer_q, sec_list, first_layer_input_q, layer_prop, batch_adj_prop, max(first_layer_addition_q,
                                                                                         second_layer_addition_q)


# Main part

ATOL_DEFAULT = 1e-8
RTOL_DEFAULT = 1e-5
def is_identity_matrix(mat,
                       ignore_phase=False,
                       rtol=RTOL_DEFAULT,
                       atol=ATOL_DEFAULT):
    """Test if an array is an identity matrix."""
    if atol is None:
        atol = ATOL_DEFAULT
    if rtol is None:
        rtol = RTOL_DEFAULT
    mat = np.array(mat)
    if mat.ndim != 2:
        return False
    if ignore_phase:
        # If the matrix is equal to an identity up to a phase, we can
        # remove the phase by multiplying each entry by the complex
        # conjugate of the phase of the [0, 0] entry.
        theta = np.angle(mat[0, 0])
        mat = np.exp(-1j * theta) * mat
    # Check if square identity
    iden = np.eye(len(mat))
    return np.allclose(mat, iden, rtol=rtol, atol=atol)


def is_unitary_matrix(mat, rtol=RTOL_DEFAULT, atol=ATOL_DEFAULT):
    """Test if an array is a unitary matrix."""
    mat = np.array(mat)
    # Compute A^dagger.A and see if it is identity matrix
    mat = np.conj(mat.T).dot(mat)
    return is_identity_matrix(mat, ignore_phase=False, rtol=rtol, atol=atol)



def q_map_neural_compute_body(circ, inputs, iq, aux_qr, inference_batch_size, log_batch_size, weights,Q_InputMatrix):
    quantum_gates, ret_index = qf_map_extract_from_weight(weights)
    # print(ret_index)
    # print(quantum_gates)
    #
    expand_for_batch_index = copy.deepcopy(ret_index)
    for b in range(inference_batch_size - 1):
        start = len(ret_index) * (b + 1)
        new_batch_index = [x + start for x in ret_index]
        expand_for_batch_index += new_batch_index
    index = torch.LongTensor(expand_for_batch_index)
    Input0 = copy.deepcopy(Q_InputMatrix)
    Input0 = Input0[index]
    # print("for debug comparison")
    # print(Q_InputMatrix[:,0])

    # print(Input0[:,0])
    circ.append(UnitaryGate(Input0, label="Input0"), inputs[0:iq])

    # print(circ)
    # backend = Aer.get_backend('unitary_simulator')
    # job = execute(circ, backend)
    # result = job.result()
    # torch.set_printoptions(threshold=sys.maxsize)
    # np.set_printoptions(threshold=sys.maxsize)
    # state = result.get_unitary(circ, decimals=4)
    # print(state[:,0])
    #

    qbits = inputs[0:iq - log_batch_size]
    for gate in quantum_gates:
        z_count = gate.count("1")
        # z_pos = get_index_list(gate,"1")
        z_pos = get_index_list(gate[::-1], "1")
        # print(z_pos)
        if z_count == 1:
            circ.z(qbits[z_pos[0]])
        elif z_count == 2:
            circ.cz(qbits[z_pos[0]], qbits[z_pos[1]])
        elif z_count == 3:
            qiskit_library.ccz(circ, qbits[z_pos[0]], qbits[z_pos[1]], qbits[z_pos[2]], aux_qr[0])
        elif z_count == 4:
            qiskit_library.cccz(circ, qbits[z_pos[0]], qbits[z_pos[1]], qbits[z_pos[2]], qbits[z_pos[3]], aux_qr[0],
                                aux_qr[1])

    #
    #
    # print(circ)
    # backend = Aer.get_backend('unitary_simulator')
    # job = execute(circ, backend)
    # result = job.result()
    #
    # torch.set_printoptions(threshold=sys.maxsize)
    # np.set_printoptions(threshold=sys.maxsize)
    # state = result.get_unitary(circ, decimals=4)
    # print(state[:,0])
    # sys.exit(0)


def q_map_neural_compute_extract(circ, inputs, iq, outputs, aux_qr, log_batch_size):
    qbits = inputs[0:iq - log_batch_size]
    # qbits = inputs[log_batch_size:iq]
    for q in qbits:
        circ.h(q)

    circ.barrier()

    for q in qbits:
        circ.x(q)

    digits = log_batch_size
    flag = "0" + str(digits) + "b"
    # qbits = inputs[0:log_batch_size]
    qbits = inputs[iq - log_batch_size:iq]
    if log_batch_size != 0:
        for i in range(int(math.pow(2, log_batch_size))):
            binstr_cur_num = format(i, flag)[::-1]

            for pos in range(len(binstr_cur_num)):
                if binstr_cur_num[pos] == "0":
                    circ.x(qbits[pos])

            if digits == 1:
                qiskit_library.cccccx(circ, inputs[0:iq], outputs[i], aux_qr)
            elif digits == 2:
                qiskit_library.ccccccx(circ, inputs[0:iq], outputs[i], aux_qr)
            elif digits == 3:
                qiskit_library.cccccccx(circ, inputs[0:iq], outputs[i], aux_qr)

            for pos in range(len(binstr_cur_num)):
                if binstr_cur_num[pos] == "0":
                    circ.x(qbits[pos])
            circ.barrier()

    else:
        qiskit_library.ccccx(circ, inputs[0], inputs[1], inputs[2], inputs[3], outputs[0], aux_qr[0], aux_qr[1])

    circ.barrier()
    qbits = inputs[log_batch_size:iq]
    for q in qbits:
        circ.x(q)


def q_map_q_ori_net(circ, s_qr_in, s_qr_enc, aux_qr, weights):
    if len(weights) == 4:
        for i in range(len(weights)):
            if weights[i] == -1:
                circ.x(s_qr_in[i])
        circ.h(s_qr_enc[0])
        circ.h(s_qr_enc[1])
        circ.barrier()
        circ.x(s_qr_enc[0])
        circ.x(s_qr_enc[1])
        ccz(circ, s_qr_in[3], s_qr_enc[0], s_qr_enc[1], aux_qr[0])
        circ.x(s_qr_enc[0])
        circ.x(s_qr_enc[1])

        circ.x(s_qr_enc[1])
        ccz(circ, s_qr_in[2], s_qr_enc[0], s_qr_enc[1], aux_qr[0])
        circ.x(s_qr_enc[1])

        circ.x(s_qr_enc[0])
        ccz(circ, s_qr_in[1], s_qr_enc[0], s_qr_enc[1], aux_qr[0])
        circ.x(s_qr_enc[0])

        ccz(circ, s_qr_in[0], s_qr_enc[0], s_qr_enc[1], aux_qr[0])

        circ.barrier()
        for i in range(len(weights)):
            if weights[i] == -1:
                circ.x(s_qr_in[i])

    elif len(weights) == 8:
        print("Size", len(weights), "in 2nd layer is now not supportted")
        sys.exit(0)
    else:
        print("Size", len(weights), "in 2nd layer is now not supportted")
        sys.exit(0)


def q_map_bn(circ, s_qr_enc, s_qr_bat, output, aux_qr, enc_len, type, val):
    if enc_len != 2:
        print("Encoder size of ", enc_len, "is now not supportted")
        sys.exit(0)
    else:
        for i in range(enc_len):
            circ.h(s_qr_enc[i])
            circ.x(s_qr_enc[i])
        circ.barrier()
        circ.ccx(s_qr_enc[0], s_qr_enc[1], s_qr_bat[0])
        qc_ang = 2 * torch.tensor(math.sqrt(val)).asin().item()
        circ.ry(qc_ang, s_qr_bat[1])
        if type == 1:
            circ.ccx(s_qr_bat[0], s_qr_bat[1], output)


# def modify_target(target):
#     for j in range(len(target)):
#         for idx in range(len(interest_num)):
#             if target[j] == interest_num[idx]:
#                 target[j] = idx
#                 break
#
#     new_target = torch.zeros(target.shape[0], 2)
#
#     for i in range(target.shape[0]):
#         if target[i].item() == 0:
#             new_target[i] = torch.tensor([1, 0]).clone()
#         else:
#             new_target[i] = torch.tensor([0, 1]).clone()
#
#     return target, new_target




def modify_target_ori(target,interest_num):
    for j in range(len(target)):
        for idx in range(len(interest_num)):
            if target[j] == interest_num[idx]:
                target[j] = idx
                break

    new_target = torch.zeros(target.shape[0], len(interest_num))

    for i in range(target.shape[0]):
        one_shot = torch.zeros(len(interest_num))
        one_shot[target[i].item()] = 1
        new_target[i] = one_shot.clone()

    return target, new_target


def modify_target(target,interest_num):
    new_target = torch.zeros(target.shape[0], len(interest_num))

    for i in range(target.shape[0]):
        one_shot = torch.zeros(len(interest_num))
        one_shot[target[i].item()] = 1
        new_target[i] = one_shot.clone()
    return target, new_target




def select_num(dataset, interest_num):
    labels = dataset.targets  # get labels
    labels = labels.numpy()
    idx = {}
    for num in interest_num:
        idx[num] = np.where(labels == num)

    fin_idx = idx[interest_num[0]]
    for i in range(1, len(interest_num)):
        fin_idx = (np.concatenate((fin_idx[0], idx[interest_num[i]][0])),)

    fin_idx = fin_idx[0]

    dataset.targets = labels[fin_idx]
    dataset.data = dataset.data[fin_idx]

    # print(dataset.targets.shape)

    dataset.targets, _ = modify_target_ori(dataset.targets,interest_num)
    # print(dataset.targets.shape)

    return dataset


def qc_input_trans(dataset):
    dataset.data = dataset.data
    return dataset


class ToQuantumData(object):
    def __call__(self, tensor):
        data = tensor
        input_vec = data.view(-1)
        vec_len = input_vec.size()[0]
        input_matrix = torch.zeros(vec_len, vec_len)
        input_matrix[0] = input_vec
        input_matrix = input_matrix.transpose(0, 1)
        u, s, v = np.linalg.svd(input_matrix)
        output_matrix = torch.tensor(np.dot(u, v))
        output_data = output_matrix[:, 0].view(1, img_size, img_size)
        return output_data


class ToQuantumMatrix(object):
    def __call__(self, tensor):
        data = tensor
        input_vec = data.view(-1)
        vec_len = input_vec.size()[0]
        input_matrix = torch.zeros(vec_len, vec_len)
        input_matrix[0] = input_vec
        input_matrix = np.float64(input_matrix.transpose(0, 1))
        u, s, v = np.linalg.svd(input_matrix)
        output_matrix = torch.tensor(np.dot(u, v))
        return output_matrix


class ToQuantumData_Batch(object):
    def __call__(self, tensor):
        data = tensor
        input_vec = data.view(-1)
        vec_len = input_vec.size()[0]
        input_matrix = torch.zeros(vec_len, vec_len)
        input_matrix[0] = input_vec
        input_matrix = input_matrix.transpose(0, 1)
        u, s, v = np.linalg.svd(input_matrix)
        output_matrix = torch.tensor(np.dot(u, v))
        output_data = output_matrix[:, 0].view(data.shape)
        return output_data




def save_checkpoint(state, is_best, save_path, filename):
    filename = os.path.join(save_path, filename)
    torch.save(state, filename)
    if is_best:
        bestname = os.path.join(save_path, 'model_best.tar')
        shutil.copyfile(filename, bestname)




def analyze(counts):
    num_c_reg = 2
    mycount = {}
    for i in range(num_c_reg):
        mycount[i] = 0
    for k, v in counts.items():
        bits = len(k)
        for i in range(bits):
            if k[bits - 1 - i] == "1":
                if i in mycount.keys():
                    mycount[i] += v
                else:
                    mycount[i] = v
    return mycount, bits


def fire_ibmq(circuit, shots, iter, Simulation=False, printable=True, backend_name='ibmq_essex'):
    if printable:
        print(circuit)

    count_set = []
    start = time.time()
    for it in range(iter):
        if not Simulation:
            provider = IBMQ.get_provider('ibm-q-academic')
            # ibm-q-academic backends:
            #  5 qubits: ibmq_valencia
            # 20 qubits: ibmq_poughkeepsie, ibmq_johannesburg,ibmq_boeblingen, ibmq_20_tokyo
            # 53 qubits: ibmq_rochester

            # To get a specific qubit backend:
            backend = provider.get_backend(backend_name)
        else:
            backend = Aer.get_backend('qasm_simulator')
        job_ibm_q = execute(circuit, backend, shots=shots)
        job_monitor(job_ibm_q)
        result_ibm_q = job_ibm_q.result()
        counts = result_ibm_q.get_counts()
        count_set.append(counts)
    end = time.time()
    # print("Simulation time:", end - start)

    return count_set

def simulation_36(iq,nn_prop,bn_prop,Q_InputMatrix):
    inference_batch_size = 1
    log_batch_size =0

    f_qr = QuantumRegister(4, "fLayer")
    aux_qr = QuantumRegister(2, "aux")
    s_qr_in = QuantumRegister(1, "s_in")
    c = ClassicalRegister(1, "reg")

    FLayer_Res = []
    for i in range(nn_prop[0][1]):
        circ = QuantumCircuit(f_qr, s_qr_in, aux_qr, c)
        q_map_neural_compute_body(circ, f_qr[0:iq], iq, aux_qr, inference_batch_size, log_batch_size, nn_prop[0][2][i],Q_InputMatrix)
        circ.barrier()
        q_map_neural_compute_extract(circ, f_qr[0:iq], iq, s_qr_in[0:inference_batch_size], aux_qr, log_batch_size)
        circ.barrier()
        circ.measure(s_qr_in, c)

        iters = 1
        qc_shots = 8192
        counts = fire_ibmq(circ, qc_shots, iters, True, False)
        (mycount, bits) = analyze(counts[0])
        for b in range(bits):
            FLayer_Res.append(float(mycount[b]) / qc_shots)
    print(FLayer_Res)

    # %%


    s_qr_in = QuantumRegister(4, "s_in")
    s_qr_enc = QuantumRegister(4, "s_encoder")
    s_qr_out = QuantumRegister(2, "s_out")
    aux_qr = QuantumRegister(1, "aux")
    c = ClassicalRegister(2, "reg")

    circ = QuantumCircuit(s_qr_in, s_qr_enc, s_qr_out, aux_qr, c)
    for i in range(len(FLayer_Res)):
        ang = 2 * torch.tensor(math.sqrt(FLayer_Res[i])).asin().item()
        circ.ry(ang, s_qr_in[i])

    # Build the computation of original neural comp
    enc_len = int(math.log(nn_prop[1][0], 2))
    for i in range(nn_prop[1][1]):
        q_map_q_ori_net(circ, s_qr_in, s_qr_enc[enc_len * i:enc_len * (i + 1)], aux_qr, nn_prop[1][2][i])
        circ.barrier()
    circ.barrier()

    circ.h(s_qr_enc[0])
    circ.x(s_qr_enc[0])
    circ.h(s_qr_enc[1])
    circ.x(s_qr_enc[1])
    circ.ccx(s_qr_enc[0], s_qr_enc[1], s_qr_out[0])

    circ.barrier()
    circ.h(s_qr_enc[2])
    circ.x(s_qr_enc[2])
    circ.h(s_qr_enc[3])
    circ.x(s_qr_enc[3])
    circ.ccx(s_qr_enc[2], s_qr_enc[3], s_qr_out[1])
    circ.barrier()

    circ.measure(s_qr_out[0], c[0])
    circ.measure(s_qr_out[1], c[1])

    SLayer_Res = []
    iters = 1
    qc_shots = 8192
    counts = fire_ibmq(circ, qc_shots, iters, True, False)
    (mycount, bits) = analyze(counts[0])
    for b in range(bits):
        SLayer_Res.append(float(mycount[b]) / qc_shots)

    print(SLayer_Res)

    # %%

    s_qr_in = QuantumRegister(2, "s_in")
    s_qr_bat = QuantumRegister(2, "s_BN")
    s_qr_out = QuantumRegister(2, "s_out")
    c = ClassicalRegister(2, "reg")

    circ = QuantumCircuit(s_qr_in, s_qr_bat, s_qr_out, c)

    for i in range(len(SLayer_Res)):
        ang = 2 * torch.tensor(math.sqrt(SLayer_Res[i])).asin().item()
        circ.ry(ang, s_qr_in[i])

    if bn_prop[1]['x_l_0_5'][0] == 0:
        # upward
        val = bn_prop[1]['x_running_rot'][0]
        qc_ang = 2 * torch.tensor(math.sqrt(val)).asin().item()
        circ.ry(qc_ang, s_qr_bat[0])
        circ.ccx(s_qr_in[0], s_qr_bat[0], s_qr_out[0])
    else:
        # downward
        val = bn_prop[1]['x_running_rot'][0]
        qc_ang = 2 * torch.tensor(math.sqrt(val)).asin().item()
        circ.ry(qc_ang, s_qr_bat[0])
        circ.cx(s_qr_in[0], s_qr_out[0])
        circ.x(s_qr_in[0])
        circ.ccx(s_qr_in[0], s_qr_bat[0], s_qr_out[0])

    circ.barrier()
    if bn_prop[1]['x_l_0_5'][1] == 0:
        val = bn_prop[1]['x_running_rot'][1]
        qc_ang = 2 * torch.tensor(math.sqrt(val)).asin().item()
        circ.ry(qc_ang, s_qr_bat[1])
        circ.ccx(s_qr_in[1], s_qr_bat[1], s_qr_out[1])
    else:
        # downward
        val = bn_prop[1]['x_running_rot'][1]
        qc_ang = 2 * torch.tensor(math.sqrt(val)).asin().item()
        circ.ry(qc_ang, s_qr_bat[1])
        circ.cx(s_qr_in[1], s_qr_out[1])
        circ.x(s_qr_in[1])
        circ.ccx(s_qr_in[1], s_qr_bat[1], s_qr_out[1])

    circ.measure(s_qr_out[0], c[0])
    circ.measure(s_qr_out[1], c[1])

    Final_Res = []

    iters = 1
    qc_shots = 8192
    counts = fire_ibmq(circ, qc_shots, iters, True, False)
    (mycount, bits) = analyze(counts[0])
    for b in range(bits):
        Final_Res.append(float(mycount[b]) / qc_shots)

    print(Final_Res)
    return torch.tensor([Final_Res])




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='QuantumFlow Qiskit Simulation')
    parser.add_argument('-c','--interest_class',default="3, 6",help="investigate classes",)
    parser.add_argument('-s', '--segment', default="0, 1", help="test segment [3,6] 0->1968", )
    parser.add_argument('-r', '--resume_path', default='mnist_36_0.9746.pth.tar', help='resume from checkpoint')
    args = parser.parse_args()



    interest_num = [int(x.strip()) for x in args.interest_class.split(",")]
    segment = [int(x.strip()) for x in args.segment.split(",")]
    resume_path = args.resume_path
    img_size = 4
    # number of subprocesses to use for data loading
    num_workers = 0
    # how many samples per batch to load
    batch_size = 1
    inference_batch_size = 1

    device = torch.device("cpu")
    layers = [4, 2]


    print("="*100)
    print("Demo 3 on MNIST. This script is for batch of data generation.")
    print("\tStart at:",time.strftime("%m/%d/%Y %H:%M:%S"))
    print("\tProblems and issues, please contact Dr. Weiwen Jiang (wjiang2@nd.edu)")
    print("\tEnjoy and Good Luck!")
    print("="*100)
    print()




    # convert data to torch.FloatTensor
    transform = transforms.Compose([transforms.Resize((img_size, img_size)),
                                    transforms.ToTensor()])
    # transform = transforms.Compose([transforms.Resize((img_size,img_size)),
    #                                 transforms.ToTensor(),ToQuantumData()])
    # transform = transforms.Compose([transforms.Resize((img_size,img_size)),transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
    # choose the training and test datasets

    # Path to MNIST Dataset
    train_data = datasets.MNIST(root='../../pytorch/data', train=True,
                                download=True, transform=transform)
    test_data = datasets.MNIST(root='../../pytorch/data', train=False,
                               download=True, transform=transform)

    train_data = select_num(train_data, interest_num)
    test_data = select_num(test_data, interest_num)

    # train_data = qc_input_trans(train_data)

    # imshow(torchvision.utils.make_grid(train_data[0][0]))
    #
    # sys.exit(0)

    # prepare data loaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                               num_workers=num_workers, shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=inference_batch_size,
                                              num_workers=num_workers, shuffle=False, drop_last=True)



    # Network Architecture: 2 layers and each layer contains 2 neurons



    model = Net(img_size,layers,True,[[1,1,1,1],[1,1]],True,False,False,False,False).to(device)


    print("=> loading checkpoint from '{}'<=".format(resume_path))
    checkpoint = torch.load(resume_path, map_location=device)
    epoch_init, acc = checkpoint["epoch"], checkpoint["acc"]
    model.load_state_dict(checkpoint["state_dict"])



    for name, para in model.named_parameters():
        if "fc" in name:
            print(name,binarize(para))
        else:
            print(name, para)


    f,sec_list,iq,nn_prop,bn_prop,aux = extract_model(model)
    model.eval()

    correct = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        if batch_idx<segment[0]:
            continue
        if batch_idx>=segment[1]:
            break
        torch.set_printoptions(threshold=sys.maxsize)
        target, new_target = modify_target(target, interest_num)
        print("Iteration:", batch_idx,target)
        quantum_data = ToQuantumData_Batch()(data)
        output = model(quantum_data, False)
        print("=" * 10, "Classic:", output)


        Q_InputMatrix = ToQuantumMatrix()(data.flatten())
        output = simulation_36(iq, nn_prop, bn_prop, Q_InputMatrix)
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        print("=" * 10, "QC:", output)
        print("="*10, "Correct num:",pred.eq(target.data.view_as(pred)).cpu().sum())



    a = 100. * correct / len(test_loader.dataset)

    print('Test set: Accuracy: {}/{} ({:.2f}%)'.format(correct, segment[1]-segment[0],
        100. * float(correct) / float(segment[1]-segment[0])))

    print(output)

    print("\tEnd at:", time.strftime("%m/%d/%Y %H:%M:%S"))