# https://github.com/CleverCracker/Quantum_Image_Based_Search_Engine/blob/42baeb62714d685a011213fe209cab208154e301/SearchEngine_32x32.py
from qiskit import IBMQ, QuantumCircuit, ClassicalRegister, QuantumRegister, execute
from qiskit.tools.monitor import job_monitor
import numpy as np
import json
import time

imageNames = ["Fukuoka", "Nagoya", "Osaka",
              "Sapporo", "Tokyo", "Black", "InkedBlack", "White", "InkedWhite", "Lena"]
result = []
data = np.loadtxt('data.csv', delimiter=',', dtype=np.complex128)

targetQubit = QuantumRegister(1, 'target')
ref = QuantumRegister(11, 'ref')
original = QuantumRegister(11, 'original')
anc = QuantumRegister(1, 'anc')
c = ClassicalRegister(1)
job = []

numOfShots = 1024
IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
backend = provider.get_backend('ibmq_qasm_simulator')
index = 7

seconds = time.time()
local_time = time.ctime(seconds)
print("Local time:", local_time)
qc = QuantumCircuit(targetQubit, ref, original, anc, c)

qc.initialize(data[index], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 23])
qc.initialize(data[0], range(12, 24))

qc.tdg(targetQubit[0])
qc.h(targetQubit[0])

for i in range(len(ref)):
    qc.cswap(targetQubit[0], ref[i], original[i])

qc.h(targetQubit[0])
qc.tdg(targetQubit[0])

qc.measure(targetQubit[0], c)

job.append(execute(qc, backend, shots=numOfShots))
print(0)
qc = QuantumCircuit(targetQubit, ref, original, anc, c)

qc.initialize(data[index], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 23])
qc.initialize(data[1], range(12, 24))

qc.tdg(targetQubit[0])
qc.h(targetQubit[0])

for i in range(len(ref)):
    qc.cswap(targetQubit[0], ref[i], original[i])

qc.h(targetQubit[0])
qc.tdg(targetQubit[0])

qc.measure(targetQubit[0], c)

job.append(execute(qc, backend, shots=numOfShots))
print(1)
qc = QuantumCircuit(targetQubit, ref, original, anc, c)

qc.initialize(data[index], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 23])
qc.initialize(data[2], range(12, 24))

qc.tdg(targetQubit[0])
qc.h(targetQubit[0])

for i in range(len(ref)):
    qc.cswap(targetQubit[0], ref[i], original[i])

qc.h(targetQubit[0])
qc.tdg(targetQubit[0])

qc.measure(targetQubit[0], c)

job.append(execute(qc, backend, shots=numOfShots))
print(2)
qc = QuantumCircuit(targetQubit, ref, original, anc, c)

qc.initialize(data[index], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 23])
qc.initialize(data[3], range(12, 24))

qc.tdg(targetQubit[0])
qc.h(targetQubit[0])

for i in range(len(ref)):
    qc.cswap(targetQubit[0], ref[i], original[i])

qc.h(targetQubit[0])
qc.tdg(targetQubit[0])

qc.measure(targetQubit[0], c)

job.append(execute(qc, backend, shots=numOfShots))
print(3)
qc = QuantumCircuit(targetQubit, ref, original, anc, c)

qc.initialize(data[index], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 23])
qc.initialize(data[4], range(12, 24))

qc.tdg(targetQubit[0])
qc.h(targetQubit[0])

for i in range(len(ref)):
    qc.cswap(targetQubit[0], ref[i], original[i])

qc.h(targetQubit[0])
qc.tdg(targetQubit[0])

qc.measure(targetQubit[0], c)

job.append(execute(qc, backend, shots=numOfShots))
print(4)

job_monitor(job[0])
job_monitor(job[1])
job_monitor(job[2])
job_monitor(job[3])
job_monitor(job[4])

for x in range(5, len(data)):
    qc = QuantumCircuit(targetQubit, ref, original, anc, c)

    qc.initialize(data[5], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 23])
    qc.initialize(data[x], range(12, 24))

    qc.tdg(targetQubit[0])
    qc.h(targetQubit[0])

    for i in range(len(ref)):
        qc.cswap(targetQubit[0], ref[i], original[i])

    qc.h(targetQubit[0])
    qc.tdg(targetQubit[0])

    qc.measure(targetQubit[0], c)

    job.append(execute(qc, backend, shots=numOfShots))
    print(x)

job_monitor(job[5])
job_monitor(job[6])
job_monitor(job[7])

seconds = time.time()
local_time = time.ctime(seconds)
print("Local time After: ", local_time)
counts = []

for i in range(len(data)):
    count = job[i].result().get_counts()
    counts.append(count)
    print(count['0'] / 1024 * 100)
    samilarty_per = count['0'] / 1024 * 100

# json.dumps(counts)
with open(str(index)+"_Result.json", 'w') as f:
    json.dump(counts, f)
seconds = time.time()
local_time = time.ctime(seconds)
print("Local time Closed: ", local_time)


"""
Time For Comparing Images :

Starting Time  : 10:47:12
Ending Time    : 11:00:32
-------------------------
Execution Time = 00:13:20
-------------------------
"""
