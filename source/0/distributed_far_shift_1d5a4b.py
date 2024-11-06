# https://github.com/beaudoco/qiskit-shifting-simulator/blob/d3a1904248c6f27eb25d2cb2a753ca4bdb0eef42/distributed/distributed_far_shift.py
import openpyxl as xl
from tqdm import tqdm
import time
# import matplotlib.pyplot as plt
from qiskit import *
from copy import deepcopy

## Loading account and backend

# QX_TOKEN = "929e8951d2081e9da2d290c48fc02a9cbd264affbcfc9669a63af613b630a8d545a504adf27d70a6650bf17dfea2fae9400895cb2f0f9f2e4c68466654505723"
QX_TOKEN = "67313723797a8e1e5905db1cd035fe6918ea028b47a6ab963058182756fbfc7f6b72e92b21c668900e83e60d206de10aec97751d91ef74de7fde33f31e4b4e58"

IBMQ.enable_account(QX_TOKEN)
provider = IBMQ.get_provider(hub='ibm-q-research', group='penn-state-1', project='main')

# QX_TOKEN = "67313723797a8e1e5905db1cd035fe6918ea028b47a6ab963058182756fbfc7f6b72e92b21c668900e83e60d206de10aec97751d91ef74de7fde33f31e4b4e58"
# provider = IBMQ.enable_account(QX_TOKEN)

### Chose the backend

backend = provider.get_backend('ibmq_athens')
# backend = FakeMelbourne()
# backend = Aer.get_backend("qasm_simulator") 

## Initializing the Quantum Circuit

qr = QuantumRegister(5, 'q')
cr = ClassicalRegister(2, 'c')
qc_close = QuantumCircuit(qr, cr)
qc_far = QuantumCircuit(qr, cr)

num_gates = 20 # change this line
q1, q2 = 3, 4
for i in range(num_gates):
    # prepare close 
    qc_close.cx(qr[0], qr[1])
    qc_close.cx(qr[2], qr[3])
    qc_close.barrier()

    # prepare far 
    qc_far.cx(qr[0], qr[1])
    qc_far.cx(qr[q1], qr[q2])

    qc_far.barrier()
    
qc_close_p2 = deepcopy(qc_close)
qc_far_p2 = deepcopy(qc_far)

# measure for close
qc_close.measure(qr[0], cr[0])
qc_close.measure(qr[1], cr[1])

qc_close_p2.measure(qr[2], cr[0])
qc_close_p2.measure(qr[3], cr[1])

# measure for far
qc_far.measure(qr[0], cr[0])
qc_far.measure(qr[1], cr[1])

qc_far_p2.measure(qr[3], cr[0])
qc_far_p2.measure(qr[4], cr[1])

## Executing the Quantum Circuit on a Hardware

max_experiments = 18
circ_list = []
for i in range(max_experiments):
    circ_list.append(qc_close)
    circ_list.append(qc_close_p2)
    
    circ_list.append(qc_far)
    circ_list.append(qc_far_p2)

# print(circ_list[0])
# print(circ_list[1])
# print(circ_list[2])
# print(circ_list[3])
# exit()

job = execute(circ_list, backend, shots=4096)
result = job.result()

t = time.time()

## Save the results in an Excel File

row2 = list(result.get_counts(1).keys())
file_counts = open("distributed_shots_{}_{}_far.txt".format(num_gates, backend), "a")

if not os.path.exists("distributed_shots_far.xlsx"):
    wb = xl.Workbook()
    ws = wb.active

    row = list(result.get_counts(1).keys())
    row.insert(0, "Backend")
    row.insert(0, "Time")

    ws.append(row)
else:
    wb = xl.load_workbook(filename = "distributed_shots_far.xlsx")
    ws = wb.active
    
for k in tqdm(range(max_experiments * 4)):
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

wb.save("distributed_shots_far.xlsx")
file_counts.close()   