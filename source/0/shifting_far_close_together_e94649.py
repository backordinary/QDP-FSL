# https://github.com/beaudoco/qiskit-shifting-simulator/blob/d533079c21f7fce5b10d058953991ddf74e44e43/shifting_far_close_together.py
## Imports

# import os
# import random
# import networkx as nx 
# import collections
import openpyxl as xl
from tqdm import tqdm
import time
# import matplotlib.pyplot as plt
from qiskit import *
# from qiskit.test.mock import FakeAlmaden, FakeVigo, FakeValencia, FakeMelbourne, FakeTokyo
# from qiskit.providers.aer import QasmSimulator
# import qiskit.transpiler.coupling as coupling 
# from qiskit.providers.aer.noise import NoiseModel
# import qiskit.providers.aer.noise as noise


## Loading account and backend

QX_TOKEN = "929e8951d2081e9da2d290c48fc02a9cbd264affbcfc9669a63af613b630a8d545a504adf27d70a6650bf17dfea2fae9400895cb2f0f9f2e4c68466654505723"

IBMQ.enable_account(QX_TOKEN)
provider = IBMQ.get_provider(hub='ibm-q-research', group='penn-3', project='main')

### Chose the backend

backend = provider.get_backend('ibmq_bogota')
# backend = FakeMelbourne()

## Initializing the Quantum Circuit

qr = QuantumRegister(5, 'q')
cr = ClassicalRegister(2, 'c')
qc_close = QuantumCircuit(qr, cr)
qc_always_far = QuantumCircuit(qr, cr)
qc_far = QuantumCircuit(qr, cr)


num_gates = 4 # change this line
q1, q2 = 2, 3
for i in range(num_gates):
    # prepare close 
    qc_close.cx(qr[0], qr[1])
    qc_close.cx(qr[2], qr[3])
    qc_close.barrier()

    # prepare always far
    qc_always_far.cx(qr[0], qr[1])
    qc_always_far.cx(qr[3], qr[4])
    qc_always_far.barrier()
    
    # prepare far
    if i == num_gates//2:
        qc_far.swap(qr[3], qr[4])
        qc_far.swap(qr[2], qr[3])
        q1, q2 = 3, 4
        qc_far.barrier()

    qc_far.cx(qr[0], qr[1])
    qc_far.cx(qr[q1], qr[q2])

    qc_far.barrier()

# measure for close
qc_close.measure(qr[0], cr[0])
qc_close.measure(qr[1], cr[1])
# qc_close.measure(qr[2], cr[2])
# qc_close.measure(qr[3], cr[3])

# measure for always far
qc_always_far.measure(qr[0], cr[0])
qc_always_far.measure(qr[1], cr[1])

# measure for far
qc_far.measure(qr[0], cr[0])
qc_far.measure(qr[1], cr[1])
# qc_far.measure(qr[q1], cr[2])
# qc_far.measure(qr[q2], cr[3])




## Executing the Quantum Circuit on a Hardware

max_experiments = 75
circ_list = []
for i in range(max_experiments):
    circ_list.append(qc_close)
    circ_list.append(qc_always_far)
    circ_list.append(qc_far)

# print(circ_list[0])
# print(circ_list[1])
# print(circ_list[2])
# exit()
job = execute(circ_list, backend, shots=8192)
result = job.result()

t = time.time()

## Save the results in an Excel File

row2 = list(result.get_counts(1).keys())
file_counts = open("counts_num_gates_{}_{}_far_close.txt".format(num_gates, backend), "a")

if not os.path.exists("far_close.xlsx"):
    wb = xl.Workbook()
    ws = wb.active

    row = list(result.get_counts(1).keys())
    row.insert(0, "Backend")
    row.insert(0, "Time")

    ws.append(row)
else:
    wb = xl.load_workbook(filename = "melbourne_far.xlsx")
    ws = wb.active


for k in tqdm(range(max_experiments)):
    print("*************************************************** \
        ***************************************************")
    print("Circuit Index {} {}".format(k, backend), result.get_counts(k))
    print("*************************************************** \
        ***************************************************")

    file_counts.write("{};{};{}\n".format(t, backend, result.get_counts(k)))

    val = result.get_counts(k)
    row = []
    for i in range(len(row2)):
        row.append(val[row2[i]])
    row.insert(0, str(backend))
    row.insert(0, t)
    ws.append(row)     

wb.save("melbourne_far.xlsx")
file_counts.close()   