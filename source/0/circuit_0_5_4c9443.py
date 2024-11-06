# https://github.com/beaudoco/qiskit-shifting-simulator/blob/5348151514ed2c60619b813b7b0eaad693fd8e59/circuits/circuit_0_5.py
import openpyxl as xl
from tqdm import tqdm
import time
# import matplotlib.pyplot as plt
from qiskit import *
from copy import deepcopy

# Loading account and backend

QX_TOKEN = "67313723797a8e1e5905db1cd035fe6918ea028b47a6ab963058182756fbfc7f6b72e92b21c668900e83e60d206de10aec97751d91ef74de7fde33f31e4b4e58"

IBMQ.enable_account(QX_TOKEN)
provider = IBMQ.get_provider(
    hub='ibm-q-research', group='penn-state-1', project='main')

# QX_TOKEN = "67313723797a8e1e5905db1cd035fe6918ea028b47a6ab963058182756fbfc7f6b72e92b21c668900e83e60d206de10aec97751d91ef74de7fde33f31e4b4e58"
# provider = IBMQ.enable_account(QX_TOKEN)

# Chose the backend

backend = provider.get_backend('ibmq_manila')
# backend = FakeMelbourne()
# backend = Aer.get_backend("qasm_simulator")

# Initializing the Quantum Circuit

qr = QuantumRegister(5, 'q')
cr = ClassicalRegister(2, 'c')
qc_0 = QuantumCircuit(qr, cr)
qc_1_close = QuantumCircuit(qr, cr)
qc_2_close = QuantumCircuit(qr, cr)
qc_3_close = QuantumCircuit(qr, cr)
qc_4_close = QuantumCircuit(qr, cr)
qc_5_close = QuantumCircuit(qr, cr)
qc_6_close = QuantumCircuit(qr, cr)
qc_7 = QuantumCircuit(qr, cr)
qc_8_close = QuantumCircuit(qr, cr)
qc_9 = QuantumCircuit(qr, cr)

num_gates = 50  # change this line
q1, q2 = 2, 3
for i in range(num_gates):    

    # prepare close 0
    qc_0.cx(qr[0], qr[1])
    qc_0.barrier()

    # prepare close 1
    qc_1_close.cx(qr[0], qr[1])
    qc_1_close.cx(qr[2], qr[3])
    qc_1_close.barrier()

    # prepare close 2
    qc_2_close.cx(qr[0], qr[1])
    qc_2_close.cx(qr[2], qr[3])
    qc_2_close.barrier()

    # prepare close 3 & 4 & 8
    if i == num_gates//2:

        qc_3_close.swap(qr[3], qr[4])
        qc_3_close.swap(qr[2], qr[3])

        qc_4_close.swap(qr[3], qr[4])
        qc_4_close.swap(qr[2], qr[3])

        qc_8_close.swap(qr[3], qr[4])
        qc_8_close.swap(qr[2], qr[3])

        q1, q2 = 3, 4
        qc_3_close.barrier()
        qc_4_close.barrier()
        qc_8_close.barrier()

    qc_3_close.cx(qr[0], qr[1])
    qc_3_close.cx(qr[q1], qr[q2])
    qc_3_close.barrier()

    qc_4_close.cx(qr[0], qr[1])
    qc_4_close.cx(qr[q1], qr[q2])
    qc_4_close.barrier()

    qc_8_close.cx(qr[q1], qr[q2])
    qc_8_close.barrier()

    # prepare close 5
    qc_5_close.cx(qr[0], qr[1])
    qc_5_close.cx(qr[3], qr[4])
    qc_5_close.barrier()

    # prepare close 6
    qc_6_close.cx(qr[0], qr[1])
    qc_6_close.cx(qr[3], qr[4])
    qc_6_close.barrier()

    # prepare close 7
    qc_7.cx(qr[2], qr[3])
    qc_7.barrier()

    # prepare close 9
    qc_9.cx(qr[3], qr[4])
    qc_9.barrier()

# measure for close 0
qc_0.measure(qr[0], cr[0])
qc_0.measure(qr[1], cr[1])

# measure for close 1
qc_1_close.measure(qr[0], cr[0])
qc_1_close.measure(qr[1], cr[1])

# measure for close 2
qc_2_close.measure(qr[2], cr[0])
qc_2_close.measure(qr[3], cr[1])

# measure for close 3
qc_3_close.measure(qr[0], cr[0])
qc_3_close.measure(qr[1], cr[1])

# measure for close 4
qc_4_close.measure(qr[q1], cr[0])
qc_4_close.measure(qr[q2], cr[1])

# measure for close 5
qc_5_close.measure(qr[0], cr[0])
qc_5_close.measure(qr[1], cr[1])

# measure for close 6
qc_6_close.measure(qr[3], cr[0])
qc_6_close.measure(qr[4], cr[1])

# measure for close 7
qc_7.measure(qr[2], cr[0])
qc_7.measure(qr[3], cr[1])

# measure for close 8
qc_8_close.measure(qr[q1], cr[0])
qc_8_close.measure(qr[q2], cr[1])

# measure for close 9
qc_9.measure(qr[3], cr[0])
qc_9.measure(qr[4], cr[1])

# Executing the Quantum Circuit on a Hardware

max_experiments = 25
circ_list = []
for i in range(max_experiments):
    circ_list.append(qc_0)
    circ_list.append(qc_5_close)
    circ_list.append(qc_0)

# print(circ_list[0])
# exit()
job = execute(circ_list, backend, shots=8192)
result = job.result()

t = time.time()

# Save the results in an Excel File

row2 = list(result.get_counts(1).keys())
file_counts = open("circuit_0_5_{}_{}.txt".format(num_gates, backend), "a")

if not os.path.exists("circuit_0_5_manila.xlsx"):
    wb = xl.Workbook()
    ws = wb.active

    row = list(result.get_counts(1).keys())
    row.insert(0, "Backend")
    row.insert(0, "Time")

    ws.append(row)
else:
    wb = xl.load_workbook(filename="circuit_0_5_manila.xlsx")
    ws = wb.active

for k in tqdm(range(max_experiments * 3)):
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

wb.save("circuit_0_5_manila.xlsx")
file_counts.close()
