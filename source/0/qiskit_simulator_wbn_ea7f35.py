# https://github.com/weiwenjiang/qc_mnist/blob/c337d72c207fad7b998528e77795cbf0bf1f8fba/interfae/qiskit_simulator_wbn.py
# %%

import sys

sys.path.append("../qiskit")
sys.path.append("../pytorch")
from qiskit_library import *
from lib_util import *
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
# INPUTs
import torch
from torch import tensor

def do_slp_via_th(input_ori, w_ori):
    p = input_ori
    d = 4 * p * (1 - p)
    e = (2 * p - 1)
    # e_sq = torch.tensor(1)
    w = w_ori

    sum_of_sq = (d + e.pow(2)).sum(-1)
    sum_of_sq = sum_of_sq.unsqueeze(-1)
    sum_of_sq = sum_of_sq.expand(p.shape[0], w.shape[0])

    diag_p = torch.diag_embed(e)

    p_w = torch.matmul(w, diag_p)
    # print(diag_p)
    # print(w)
    # print(p_w)
    #
    z_p_w = torch.zeros_like(p_w)
    shft_p_w = torch.cat((p_w, z_p_w), -1)

    sum_of_cross = torch.zeros_like(p_w)
    length = p.shape[1]

    for shft in range(1, length):
        sum_of_cross += shft_p_w[:, :, 0:length] * shft_p_w[:, :, shft:length + shft]

    sum_of_cross = sum_of_cross.sum(-1)

    # print(sum_of_sq,sum_of_cross)
    return (sum_of_sq + 2 * sum_of_cross) / (length ** 2)


def simulate_one_step(I, W, qca_x_running_rot, qca_x_l_0_5, qc_x_running_rot, test_L1):
    qca_ang = torch.tensor([1 - (qca_x_running_rot * 2)]).acos().item()
    qc_ang = torch.tensor([1 - (qc_x_running_rot * 2)]).acos().item()


    if test_L1:
        W1 = W
        IFM = I.clone().detach()
        input = IFM[0] * 2 - 1

        q_in = qk.QuantumRegister(16, "io")
        q_enc = qk.QuantumRegister(4, "encoder")
        q_out = qk.QuantumRegister(len(W1), "output")
        c = qk.ClassicalRegister(len(W1), "reg")
        aux = qk.QuantumRegister(3, "aux")

        maxIndex = len(input)
        circuit = qk.QuantumCircuit(q_in, q_enc, q_out, aux, c)

        # print(input)

        for idx in range(len(W1)):
            SLP_16_encoding(circuit, q_in, q_enc, input, aux)
            SLP_16_Uw(circuit, q_enc, W1[idx], aux)
            circuit.barrier()

            for qbit in q_enc[0:4]:
                circuit.h(qbit)
                circuit.x(qbit)
            ccccx(circuit, q_enc[0], q_enc[1], q_enc[2], q_enc[3], q_out[idx], aux[0], aux[1])
            circuit.barrier()


            # reset_qbits(circuit,q_in)
            # reset_qbits(circuit,q_enc)

        q_qca_out = qk.QuantumRegister(1, "qca_out")
        q_qca_para = qk.QuantumRegister(1, "qca_para")
        circuit.add_register(q_qca_out)
        circuit.add_register(q_qca_para)
        if qca_x_l_0_5==1:
            circuit.ry(qca_ang, q_qca_para)
            circuit.cx(q_out,q_qca_out)
            circuit.x(q_out)
            circuit.ccx(q_out,q_qca_para,q_qca_out)
        else:
            circuit.ry(qca_ang, q_qca_para)
            circuit.cx(q_out, q_qca_out)
            circuit.x(q_qca_para)
            circuit.ccx(q_out, q_qca_para, q_qca_out)

        q_qc_out = qk.QuantumRegister(1, "qc_out")
        q_qc_para = qk.QuantumRegister(1, "qc_para")
        circuit.add_register(q_qc_out)
        circuit.add_register(q_qc_para)
        circuit.ry(qc_ang, q_qc_para)
        circuit.cx(q_qca_out, q_qc_out)
        circuit.x(q_qc_para)
        circuit.ccx(q_qca_out, q_qc_para, q_qc_out)


        for idx in range(len(W1)):
            circuit.measure(q_qc_out[idx], c[idx])


    else:
        W2 = W
        OFM1_QC = I
        input = 1 - OFM1_QC[0] * 2

        if OFM1_QC.shape[1]==4:
            q_in = qk.QuantumRegister(4, "io")
            q_enc = qk.QuantumRegister(2, "encoder")
            q_out = qk.QuantumRegister(len(W2), "output")
            c = qk.ClassicalRegister(len(W2), "reg")
            aux = qk.QuantumRegister(1, "aux")
        elif OFM1_QC.shape[1]==8:
            q_in = qk.QuantumRegister(8, "io")
            q_enc = qk.QuantumRegister(3, "encoder")
            q_out = qk.QuantumRegister(len(W2), "output")
            c = qk.ClassicalRegister(len(W2), "reg")
            aux = qk.QuantumRegister(2, "aux")

        maxIndex = len(input)
        circuit = qk.QuantumCircuit(q_in, q_enc, q_out, aux, c)

        for idx in range(len(W2)):
            if OFM1_QC.shape[1]==4:
                SLP_4_encoding(circuit, q_in, q_enc, input, aux)
                SLP_4_Uw(circuit, q_enc, W2[idx], aux)
                circuit.barrier()
                for qbit in q_enc[0:2]:
                    circuit.h(qbit)
                    circuit.x(qbit)
                circuit.ccx(q_enc[0], q_enc[1], q_out[idx])
                circuit.barrier()
            elif OFM1_QC.shape[1] == 8:
                SLP_8_encoding(circuit, q_in, q_enc, input, aux)
                SLP_8_Uw(circuit, q_enc, W2[idx], aux)
                circuit.barrier()
                for qbit in q_enc[0:3]:
                    circuit.h(qbit)
                    circuit.x(qbit)
                cccx(circuit,q_enc[0], q_enc[1], q_enc[2], q_out[idx], aux[0])
                circuit.barrier()

            # reset_qbits(circuit,q_in)
            # reset_qbits(circuit,q_enc)

        q_qca_out = qk.QuantumRegister(1, "qca_out")
        q_qca_para = qk.QuantumRegister(1, "qca_para")
        circuit.add_register(q_qca_out)
        circuit.add_register(q_qca_para)
        if qca_x_l_0_5 == 1:
            circuit.ry(qca_ang, q_qca_para)
            circuit.cx(q_out, q_qca_out)
            circuit.x(q_out)
            circuit.ccx(q_out, q_qca_para, q_qca_out)
        else:
            circuit.ry(qca_ang, q_qca_para)
            circuit.cx(q_out, q_qca_out)
            circuit.x(q_qca_para)
            circuit.ccx(q_out, q_qca_para, q_qca_out)

        q_qc_out = qk.QuantumRegister(1, "qc_out")
        q_qc_para = qk.QuantumRegister(1, "qc_para")
        circuit.add_register(q_qc_out)
        circuit.add_register(q_qc_para)
        circuit.ry(qc_ang, q_qc_para)
        circuit.cx(q_qca_out, q_qc_out)
        circuit.x(q_qc_para)
        circuit.ccx(q_qca_out, q_qc_para, q_qc_out)


        for idx in range(len(W2)):
            circuit.measure(q_qc_out[idx], c[idx])

    print(circuit)
    sys.exit(0)
    qc_shots = 8192
    num_c_reg = 1

    start = time.time()
    iters = 1
    counts = simulate(circuit, qc_shots, iters, False)

    end = time.time()
    qc_time = end - start

    # print("From QC:", counts)
    # print("Simulation elasped time:", qc_time)

    def analyze(counts):
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

    # for k,v in counts[0].items():
    #     print(k,v)
    (mycount, bits) = analyze(counts[0])

    # for b in range(bits):
    #     print(b, float(mycount[b]) / qc_shots)
    return float(mycount[0]) / qc_shots



def run_simulator(model,IFM,layers):

    IFM = IFM.view(1, -1, )

    for name, para in model.named_parameters():
        if name=="fc0.weight":
            fc0_weight = binarize(para)
        elif name=="fc1.weight":
            fc1_weight = binarize(para)
        elif name=="qc0.x_running_rot":
            qc0_x_running_rot = para
        elif name == "qca0.x_l_0_5":
            qca0_x_l_0_5 = para
        elif name == "qca0.x_running_rot":
            qca0_x_running_rot = para
        elif name == "qc1.x_running_rot":
            qc1_x_running_rot = para
        elif name == "qca1.x_running_rot":
            qca1_x_running_rot = para
        elif name == "qca1.x_l_0_5":
            qca1_x_l_0_5 = para


    # print(IFM)
    # print(fc0_weight)
    # print(qc0_x_running_rot)
    # print(qca0_x_l_0_5)
    # print(qca0_x_running_rot)
    #
    # print(fc1_weight)
    # print(qc1_x_running_rot)
    # print(qca1_x_l_0_5)
    # print(qca1_x_running_rot)
    # # sys.exit(0)

    W1 = fc0_weight
    W2 = fc1_weight

    OFM1_QC = torch.zeros([1,layers[0]])
    OFM2_QC = torch.zeros([1,layers[1]])


    idx = 0
    for w in W1:
        w = w.unsqueeze(0)

        # print("-"*100)
        # print("\t\tInputs:", IFM)
        # print("\t\tWeights:", w)
        # print("\t\tBN:", qca0_x_running_rot[idx], qca0_x_l_0_5[idx], qc0_x_running_rot[idx])
        OFM1_QC[0][idx] = simulate_one_step(IFM, w, qca0_x_running_rot[idx], qca0_x_l_0_5[idx], qc0_x_running_rot[idx], True)
        # print("\t\tResults OFM 1",idx,":", OFM1_QC[0][idx])
        idx += 1


    idx = 0
    for w in W2:
        w = w.unsqueeze(0)

        # print("-"*100)
        # print("\t\tInputs:", OFM1_QC)
        # print("\t\tWeights:", w)
        # print("\t\tBN:", qca1_x_running_rot[idx], qca1_x_l_0_5[idx], qc1_x_running_rot[idx])
        OFM2_QC[0][idx] = simulate_one_step(OFM1_QC, w, qca1_x_running_rot[idx], qca1_x_l_0_5[idx],
                                            qc1_x_running_rot[idx], False)
        # print("\t\tResults OFM 2",idx,":", OFM2_QC[0][idx])
        idx += 1

    # print("\t",OFM1_QC)
    # print("\t",OFM2_QC)

    return OFM2_QC
    # OFM = {}
    # input_data = IFM.view(1,-1, )
    # layer_idx = 0
    # for layer_name, layer in model.named_modules():
    #     if isinstance(layer, BinaryLinear):
    #         W = (binarize(model.state_dict()[layer_name+".weight"]))
    #         OFM_QC = torch.zeros((1,layer.out_features))
    #
    #         idx = 0
    #         for w in W:
    #             w = w.unsqueeze(0)
    #             # print("\t\tInputs:",input_data)
    #             # print("\t\tWeights:", w)
    #             OFM_QC[0][idx] = simulate_one_step(input_data, w, layer_idx==0)
    #             # print("\t\tResults:",OFM_QC[0][idx])
    #             idx += 1
    #
    #
    #
    #         if layer.out_features == 1:
    #             OFM_QC = torch.cat((OFM_QC, 1 - OFM_QC), -1)
    #
    #         # print("="*100)
    #         # print(layer,OFM_QC)
    #
    #         OFM[layer_name] = OFM_QC
    #         input_data = OFM_QC
    #         layer_idx += 1
    #
    # # print(OFM["fc2"])
    # return OFM["fc2"]
    #
    # idx = 0
    # for w in W1:
    #     w = w.unsqueeze(0)
    #     OFM1_QC[0][idx] = simulate_one_step(IFM, w, True)
    #     idx += 1
    #
    # idx = 0
    # for w in W2:
    #     w = w.unsqueeze(0)
    #     OFM2_QC[0][idx] = simulate_one_step(OFM1_QC, w, False)
    #     idx += 1


if __name__ == "__main__":


    res = tensor([0])
    IFM = tensor([[0.0118, 0.2039, 0.1765, 0.0000, 0.0078, 0.3098, 0.3961, 0.0078, 0.0039,
         0.1373, 0.4784, 0.0431, 0.0118, 0.3059, 0.2157, 0.0039]])
    # fc0_weight = tensor([[-1., 1., -1., -1., -1., 1., 1., -1., -1., -1., 1., -1., -1., 1.,
    #          1., -1.],
    #         [1., -1., 1., 1., 1., -1., -1., 1., 1., 1., -1., 1., 1., -1.,
    #          -1., 1.],
    #         [1., 1., 1., 1., 1., 1., -1., 1., 1., 1., 1., 1., 1., 1.,
    #          1., 1.],
    #         [1., 1., 1., 1., 1., 1., -1., 1., 1., 1., 1., 1., 1., -1.,
    #          1., 1.],
    #         [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
    #          1., 1.],
    #         [-1., 1., -1., -1., -1., 1., 1., -1., -1., -1., 1., -1., -1., 1.,
    #          1., -1.],
    #         [-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
    #          -1., -1.],
    #         [1., 1., 1., 1., 1., -1., 1., 1., 1., -1., -1., 1., 1., 1.,
    #          1., 1.]])
    # fc1_weight = tensor([[1., 1., 1., 1., -1., 1., -1., 1.],
    #         [1., 1., -1., -1., -1., 1., -1., -1.],
    #         [1., 1., 1., 1., 1., 1., 1., -1.]])
    #
    # qc0_x_running_rot = tensor([9.4118e-01, 9.4118e-01, 7.9998e-04, 8.1365e-04, 9.4131e-01, 9.4118e-01,
    #         9.4131e-01, 9.4119e-01])
    #
    # qca0_x_running_rot = tensor([0.4263, 0.4263, 0.8742, 0.0976, 0.7767, 0.4263, 0.7703, 0.1914])
    # qca0_x_l_0_5 = tensor([1., 1., 0., 1., 0., 1., 0., 1.])
    #
    # qc1_x_running_rot = ([0.9412, 0.9412, 0.9412])
    #
    # qca1_x_running_rot = tensor([0.4015, 0.4059, 0.3963])
    # qca1_x_l_0_5 = tensor([1., 1., 1.])
    #
    # out_qc0 = tensor([[5.1840e-01, 5.1840e-01, 3.4541e-04, 3.9502e-04, 3.8762e-01, 5.1840e-01,
    #      3.8441e-01, 4.4397e-01]])
    # out_qc1 = tensor([[0.4507, 0.4960, 0.4695]])

    fc0_weight = tensor([[ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1., -1.,  1.,  1.,  1.,
          1.,  1.],
        [ 1.,  1.,  1.,  1.,  1., -1.,  1.,  1.,  1., -1.,  1.,  1.,  1.,  1.,
          1.,  1.],
        [ 1., -1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1., -1.,  1.,  1.,  1.,
          1.,  1.],
        [-1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,  1., -1., -1., -1.,
         -1., -1.]])

    qca0_x_running_rot = tensor([0.0246, 0.1704, 0.2078, 0.0246])
    qca0_x_l_0_5 = tensor([1., 1., 1., 1.])
    # qca0_x_g_0_5 = tensor([1., 0., 1., 0.])
    qc0_x_running_rot = tensor([0.0143, 0.9697, 0.9697, 0.0127])

    fc1_weight = tensor([[ 1.,  1., -1.,  1.],
        [ 1., -1.,  1.,  1.]])

    qca1_x_running_rot  = tensor([0.2053, 0.2052])
    qca1_x_l_0_5 = tensor([1., 1.])
    # qca1_x_g_0_5 = tensor([0., 0.])

    qc1_x_running_rot = tensor([0.9697, 0.9697])

    IFM = tensor([[0.0000, 0.1608, 0.2549, 0.0118, 0.0000, 0.2314, 0.4314, 0.0118, 0.0235,
         0.2549, 0.4392, 0.0235, 0.0392, 0.2510, 0.1843, 0.0039]])
    # out_fc0 = tensor([[0.5922, 0.4236, 0.5922, 0.4348]])
    # out_qca0 = tensor([[0.5706, 0.5415, 0.5706, 0.4917]])
    out_qc0 = tensor([[0.0075, 0.4576, 0.5068, 0.0066]])

    # out_fc1 = tensor([[0.2997, 0.2736]])
    # out_qca1 = tensor([[0.5035, 0.4844]])
    out_qc1 = tensor([[0.5041, 0.4667]])

    W1 = fc0_weight

    # OFM1 = tensor([[0.0384, 0.3188, 0.4043, 0.0272]])

    W2 = fc1_weight

    # OFM2 = tensor([[0.3904, 0.3105]])


    OFM1_QC = torch.zeros_like(out_qc0)
    OFM2_QC = torch.zeros_like(out_qc1)




    idx = 0
    for w in W1:
        w = w.unsqueeze(0)

        print("-"*100)
        print("\t\tInputs:", IFM)
        print("\t\tWeights:", w)
        print("\t\tBN:", qca0_x_running_rot[idx], qca0_x_l_0_5[idx], qc0_x_running_rot[idx])
        OFM1_QC[0][idx] = simulate_one_step(IFM, w, qca0_x_running_rot[idx], qca0_x_l_0_5[idx], qc0_x_running_rot[idx], True)
        print("\t\tResults:", OFM1_QC[0][idx])
        idx += 1



    idx = 0
    for w in W2:
        w = w.unsqueeze(0)
        OFM2_QC[0][idx] = simulate_one_step(OFM1_QC, w, qca1_x_running_rot[idx], qca1_x_l_0_5[idx], qc1_x_running_rot[idx], False)
        idx += 1

    print(OFM1_QC)
    print(OFM2_QC)
