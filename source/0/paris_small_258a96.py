# https://github.com/weiwenjiang/qc_mnist/blob/c337d72c207fad7b998528e77795cbf0bf1f8fba/interfae/paris_small.py
# %%

import sys

sys.path.append("../qiskit")
sys.path.append("../pytorch")
from qiskit_library import *
# from mnist import *
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
import torch
import numpy as np
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from lib_model_summary import summary
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np
import math
import os
import shutil
import os
import time
import sys
from pathlib import Path
import functools
from QC_SELF import Net
from collections import Counter

from qiskit import IBMQ

# IBMQ.delete_accounts()
IBMQ.save_account(
    '62d0e14364f490e45b5b5e0f6eebdbc083270ffffb660c7054219b15c7ce99ab4aa3b321309c0a9d0c3fc20086baece1376297dcdb67c7b715f9de1e4fa79efb')
IBMQ.load_account()


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
        # job_monitor(job_ibm_q)
        result_ibm_q = job_ibm_q.result()
        counts = result_ibm_q.get_counts()
        count_set.append(counts)
    end = time.time()
    print("Simulation time:", end - start)

    return count_set





def opt_model(input):
    trans_input = []
    for idx in range(len(input)):
        y_v = input[idx].item()
        if y_v > 0:
            alpha = np.arccos(y_v)
        elif y_v < 0:
            alpha = np.pi - np.arccos(-y_v)
        else:
            alpha = np.pi / 2
        trans_input.append(alpha)

    q = qk.QuantumRegister(15, "q")
    c = qk.ClassicalRegister(2, "c")

    circuit = qk.QuantumCircuit(q, c)

    circuit.ry(trans_input[0], q[0])
    circuit.ry(trans_input[1], q[1])
    circuit.ry(1.7412, q[3])
    circuit.ry(trans_input[0], q[4])
    circuit.ry(trans_input[1], q[5])
    circuit.ry(2.3766, q[7])
    circuit.ry(1.5956, q[9])

    circuit.h(q[1])
    circuit.h(q[5])
    circuit.x(q[7])
    circuit.cz(q[0], q[1])
    circuit.cz(q[4], q[5])
    circuit.z(q[1])
    circuit.z(q[5])
    circuit.h(q[1])
    circuit.h(q[5])
    circuit.x(q[1])
    circuit.x(q[5])
    circuit.cx(q[1], q[2])
    circuit.cx(q[5], q[6])
    circuit.ccx(q[1], q[3], q[2])
    circuit.x(q[5])
    circuit.ccx(q[5], q[7], q[6])
    circuit.cx(q[6], q[8])
    circuit.ccx(q[6], q[9], q[8])
    circuit.barrier(q[7], q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[8], q[9], q[10], q[11], q[12], q[13], q[14])
    circuit.cx(q[2], q[8])
    circuit.h(q[10])
    circuit.h(q[11])
    circuit.ry(2.2800, q[13])
    circuit.ccx(q[1], q[3], q[2])
    circuit.cz(q[8], q[10])
    circuit.x(q[13])
    circuit.cx(q[1], q[2])
    circuit.cz(q[8], q[11])
    circuit.ry(0.8632, q[2])
    circuit.z(q[10])
    circuit.h(q[11])
    circuit.h(q[10])
    circuit.x(q[11])
    circuit.x(q[10])
    circuit.cx(q[10], q[12])
    circuit.x(q[10])
    circuit.ccx(q[10], q[13], q[12])
    circuit.cx(q[11], q[14])
    circuit.ccx(q[11], q[2], q[14])
    circuit.barrier(q[7], q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[8], q[9], q[10], q[11], q[12], q[13], q[14])
    circuit.measure(q[12], c[0])
    circuit.measure(q[14], c[1])

    return circuit




import random

# INPUTs
import torch
from torch import tensor

milestones = [6, 10, 14]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)

optimizer = torch.optim.Adam([
    {'params': model.fc1.parameters()},
    {'params': model.fc2.parameters()},
    {'params': model.qc1.parameters(), 'lr': 1},
    {'params': model.qc2.parameters(), 'lr': 1},
], lr=1)

scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [0, 1], gamma=0.1)

resume_path = "../pytorch/model/ipykernel_launcher.py_2020_05_04-14_22_13/checkpoint_0_0.9714.pth.tar"
if os.path.isfile(resume_path):
    print("=> loading checkpoint from '{}'<=".format(resume_path))
    checkpoint = torch.load(resume_path, map_location=device)
    epoch_init, acc = checkpoint["epoch"], checkpoint["acc"]
    model.load_state_dict(checkpoint["state_dict"])

    scheduler.load_state_dict(checkpoint["scheduler"])
    scheduler.milestones = Counter(milestones)
    optimizer.load_state_dict(checkpoint["optimizer"])
else:
    epoch_init, acc = 0, 0

# print("====")
# print(model(torch.tensor([0.95, 0.95]), False).data)
# print("====")

# sys.exit(0)
x_set = [x / 20.0 for x in range(1, 20)]

results = []
fh = open("ibmq_rochester_small.res", "w")
for i in range(100):

    x = random.choice(x_set)
    y = random.choice(x_set)

    circuit = opt_model(torch.tensor([x, y]) * 2 - 1)
    print("***************Start:", [x, y], "*****************")

    theoretic_res = model(torch.tensor([x,y]), False).data

    # qc_shots = 10000
    # num_c_reg = 2
    # print("="*50)
    # print("Start simulation:")
    # start = time.time()
    # iters = 1
    # counts = fire_ibmq(circuit,qc_shots,iters,True,False)
    # end = time.time()
    # qc_time = end - start
    #
    # (mycount,bits) = analyze(counts[0])
    #
    # res = {}
    # for b in range(bits):
    #     print (b,float(mycount[b])/qc_shots)
    #     res[b] = float(mycount[b])/qc_shots
    # print("From QC:",counts)
    # print("Simulation elasped time:",qc_time)
    # qc_res = res

    qc_shots = 8192
    num_c_reg = 2
    print("=" * 50)
    print("Start run:")
    start = time.time()
    iters = 1
    counts = fire_ibmq(circuit, qc_shots, iters, False, False, backend_name="ibmq_paris")
    end = time.time()
    qc_time = end - start

    qc_res = {}
    (mycount, bits) = analyze(counts[0])
    for b in range(bits):
        print(b, float(mycount[b]) / qc_shots)
        qc_res[b] = float(mycount[b]) / qc_shots


    print("From QC:", counts)
    print("Simulation elasped time:", qc_time)

    results.append([(x, y), qc_res, theoretic_res])
    print([(x, y), qc_res, theoretic_res], file=fh, flush=True)

    if i % 5 == 0:
        print("=" * 10, "PRINT RESULTS", "=" * 10)
        print("-" * 40)
        for k in results:
            print(k)
        print("-" * 40)


