# https://github.com/beaudoco/qiskit-shifting-simulator/blob/ade67cd57998f8a70d28fdb7452ed6beac4858a6/shifting_simulator.py
## Imports

from qiskit import *
import argparse
from qiskit.test.mock import FakeAlmaden, FakeVigo, FakeValencia, FakeMelbourne, FakeTokyo
#from qiskit.providers.aer import QasmSimulator
from qiskit import Aer
import os
import random
import qiskit.transpiler.coupling as coupling 
#from qiskit.providers.aer.noise import NoiseModel
#import qiskit.providers.aer.noise as noise 
import matplotlib.pyplot as plt
from qiskit.test.mock import FakeVigo, FakeMelbourne, FakeAlmaden, FakeValencia, FakeTokyo, FakeMelbourne
import networkx as nx 
import collections
import random
import openpyxl as xl
from tqdm import tqdm
import time

## Loading account and backend

QX_TOKEN = "67313723797a8e1e5905db1cd035fe6918ea028b47a6ab963058182756fbfc7f6b72e92b21c668900e83e60d206de10aec97751d91ef74de7fde33f31e4b4e58"
provider = IBMQ.enable_account(QX_TOKEN)

## Get Coupling map of Melbourne

backend = FakeMelbourne()
edges = backend.configuration().coupling_map

### Chose the backend
#backend = Aer.get_backend("qasm_simulator")
backend = provider.get_backend('ibmq_qasm_simulator')
# backend = FakeMelbourne()

## Initializing the Quantum Circuit

qr = QuantumRegister(14, 'q')
# anc = QuantumRegister(2, 'ancilla')
cr = ClassicalRegister(4, 'c')
qc = QuantumCircuit(qr, cr)

# qc.x(qr[2])
# qc.x(qr[11])

for i in range(50):
    qc.cx(qr[2], qr[12])
    qc.barrier(qr[2], qr[12])
    
qc.measure(qr[2], cr[0])
qc.measure(qr[12], cr[1])

for i in range(50):
    qc.cx(qr[3], qr[11])
    qc.barrier(qr[3], qr[11])

qc.measure(qr[3], cr[2])
qc.measure(qr[11], cr[3])

qc.draw(output='mpl')

## Executing the Quantum Circuit on a Hardware

max_experiments = 10
circ_list = [qc for i in range(max_experiments)]
# job = execute(circ_list, backend, shots=1024)
job = execute(circ_list, backend, shots=8192, coupling_map=edges)
result = job.result()

t = time.time()

## Save the results in an Excel File

row2 = list(result.get_counts(1).keys())
#file_counts = open("C:/Users/bring/OneDrive/Documents/penn-state/cross-talk/counts_{}_close.txt".format(backend), "a")

if not os.path.exists("qasm_close.xlsx"):
    wb = xl.Workbook()
    ws = wb.active

    row = list(result.get_counts(1).keys())
    row.insert(0, "Backend")
    row.insert(0, "Time")

    ws.append(row)
else:
    wb = xl.load_workbook(filename = "qasm_close.xlsx")
    ws = wb.active


for k in tqdm(range(max_experiments)):
    print("*************************************************** \
        ***************************************************")
    print("Circuit Index {} {}".format(k, backend), result.get_counts(k))
    print("*************************************************** \
        ***************************************************")

    #file_counts.write("{};{};{}\n".format(t, backend, result.get_counts(k)))

    val = result.get_counts(k)
    row = []
    for i in range(len(row2)):
        row.append(val[row2[i]])
    row.insert(0, str(backend))
    row.insert(0, t)
    ws.append(row)     

wb.save("qasm_close.xlsx")
#file_counts.close()   