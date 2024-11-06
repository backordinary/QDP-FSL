# https://github.com/weiwenjiang/qc_mnist/blob/c337d72c207fad7b998528e77795cbf0bf1f8fba/interfae/cambridge_5qbit.py
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
from one_layer import Net
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


def fire_ibmq(circuit, shots, iter, Simulation=False, printable=True, backend_name='ibmq_essex', qr=[]):
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

        if len(qr) != 0:
            mapping = {qr[0]: 12, qr[1]: 11, qr[2]: 10, qr[3]: 17, qr[4]: 23}
            job_ibm_q = execute(circuit, backend, shots=shots, initial_layout=mapping)
        else:
            job_ibm_q = execute(circuit, backend, shots=shots)

        # result_local = self._local_backend.run(qobj, validate_qobj=True).result()

        # job_monitor(job_ibm_q)
        result_ibm_q = job_ibm_q.result()
        counts = result_ibm_q.get_counts()
        count_set.append(counts)
    end = time.time()
    print("Simulation time:", end - start)

    return count_set


def one_layer_opt_model(input):
    x = input[0]
    y = input[1]
    input_angle = torch.tensor([((x * (1 - y) + y * (1 - x)) * 2) - 1]).acos()

    q = qk.QuantumRegister(5, "q")
    c = qk.ClassicalRegister(1, "c")

    circuit = qk.QuantumCircuit(q, c)

    circuit.ry(input_angle[0].data.item(), q[0])
    circuit.ry(0.7650, q[2])
    circuit.ry(0.3488, q[4])
    circuit.x(q[0])
    circuit.h(q[0])
    circuit.z(q[0])
    circuit.h(q[0])
    circuit.ccx(q[0], q[2], q[1])
    circuit.x(q[0])
    circuit.cx(q[0], q[1])
    circuit.cx(q[1], q[3])
    circuit.ccx(q[1], q[4], q[3])
    circuit.measure(q[3], c)
    return (circuit,q)






import random

# INPUTs
import torch
from torch import tensor

milestones = [6, 10, 14]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)

optimizer = torch.optim.Adam([
    {'params': model.fc1.parameters()},
    {'params': model.qc1.parameters(), 'lr': 1},
], lr=1)

scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [0, 1], gamma=0.1)

resume_path = "../pytorch/model/ipykernel_launcher.py_2020_05_05-20_06_18/checkpoint_1_1.0.pth.tar"
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
x_set = [x / 10.0 for x in range(1, 10)]
x_set_len = len(x_set)
results = []
fh = open("results/ibmq_cambridge_5qbit.res", "w")
for i in range(x_set_len):
    for j in range(i,x_set_len):

        x = x_set[i]
        y = x_set[j]

        circuit,qr = one_layer_opt_model(torch.tensor([x, y]))


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
        counts = fire_ibmq(circuit, qc_shots, iters, False, False, backend_name="ibmq_cambridge", qr=qr)
        end = time.time()
        qc_time = end - start

        qc_res = {}
        (mycount, bits) = analyze(counts[0])
        for b in range(bits):
            print(b, float(mycount[b]) / qc_shots)
            qc_res[b] = float(mycount[b]) / qc_shots


        print("From QC:", counts)
        print("QC elasped time:", qc_time)

        results.append([(x, y), qc_res, theoretic_res])
        print([(x, y), qc_res, theoretic_res], file=fh, flush=True)

        if i % 5 == 0:
            print("=" * 10, "PRINT RESULTS", "=" * 10)
            print("-" * 40)
            for k in results:
                print(k)
            print("-" * 40)

