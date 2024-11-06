# https://github.com/PWJ1900/QAOAPaperUse/blob/d5e08da56f170847716a2a0e79b59b3ede8e8eac/alTest.py
import time


edges = [[3, 2], [3, 4], [2, 1], [1, 4], [4, 0]] # 这里为测试的图
arr = [[0, 4], [4, 1], [1, 2], [2, 3]] #这里为优化的边， 可从dijkstraOp自动获取构造
nqubits = 5
arr2 = []
for i in arr:
    arr2.append([i[0], i[1], 1])
for i in edges:
    if [i[0], i[1]] not in arr and [i[1], i[0]] not in arr:
        arr2.append([i[0], i[1], 0])
print(arr2)

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, transpile
from qiskit import Aer, execute

fs = []


def maxcut_obj(x):
    obj = 0
    for i, j, k in arr2:
        if x[i] != x[j]:
            obj -= 1
    return obj


def compute_expectation(counts):
    avg = 0
    sum_count = 0
    for bitstring, count in counts.items():
        obj = maxcut_obj(bitstring)
        avg += obj * count
        sum_count += count
    print(avg)
    fs.append(1)
    print(len(fs))
    return avg / sum_count


def create_qaoa_circ(theta):
    nqubits = 5
    p = len(theta) // 2  # number of alternating unitaries
    qc = QuantumCircuit(nqubits)
    beta = theta[:p]
    gamma = theta[p:]
    # initial_state
    for i in range(0, nqubits):
        qc.h(i)
    for irep in range(0, p):
        # problem unitary
        for pair in list(arr2):
            if pair[2] == 0:
                qc.cx(pair[0], pair[1])
            qc.rz(2 * gamma[irep], qubit=pair[1])
            qc.cx(pair[0], pair[1])
        # mixer unitary
        for i in range(0, nqubits):
            qc.rx(2 * beta[irep], i)
    qc.measure_all()
    return qc


def get_expectation(p, shots=512):
    backend = Aer.get_backend('qasm_simulator')
    backend.shots = shots

    def execute_circ(theta):
        qc = create_qaoa_circ(theta)
        counts = backend.run(qc, seed_simulator=10,
                             nshots=512).result().get_counts()
        return compute_expectation(counts)

    return execute_circ

expectation = get_expectation(p=1)
print(expectation)

backend = Aer.get_backend('qasm_simulator')
backend.shots = 512
time_start = time.time()
qc_res = create_qaoa_circ([2.14765, 1.1066])
time_end = time.time()
print(time_end - time_start)
print(qc_res)


def defineSuce(key):
    a = []
    for i in key:
        a.append(i)
    for i in range(len(a)):
        if i < len(a) - 1 and a[i] == a[i + 1]:
            return False
    return True

def useNoiseTimes():
    counts = backend.run(qc_res, seed_simulator=10).result().get_counts()
    a = 0
    print(counts)
    # 计算值
    for i in counts:
        a = counts[i] + a
    print(a)
    b = 0
    for key, value in counts.items():
        if defineSuce(key):
            b += value / 1024
    print(b)
useNoiseTimes()
