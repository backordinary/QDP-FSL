# https://github.com/PWJ1900/QAOA-Max-cut-PaperUse/blob/cbab8cc9cc51ac7fd67d6e8e05785d42fa4b4297/circuitUse.py
import time
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, transpile
from qiskit import Aer, execute



def useQ(nbits):
    edgesTo = []
    for i in range(nbits):
        if i < nbits - 1:
            edgesTo.append((i, i+1))
        if i == nbits - 1:
            edgesTo.append((nbits - 1, 0))
    return edgesTo






nqubits = 20000
edges = useQ(nqubits)


def maxcut_obj(x):
    obj = 0
    for i, j in edges:
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
    return avg / sum_count


def create_qaoa_circ(theta):
    nqubits = 20000
    p = len(theta) // 2
    qc = QuantumCircuit(nqubits)
    beta = theta[:p]
    gamma = theta[p:]
    # initial_state
    for i in range(0, nqubits):
        qc.h(i)
    for irep in range(0, p):
        # problem unitary
        for pair in list(edges):
            if irep == 0:
                if pair == [nqubits - 1, 0]:
                    qc.cx(pair[0], pair[1])
            else:
                qc.cx(pair[0], pair[1])
            qc.rz(2 * gamma[irep], qubit=pair[1])
            qc.cx(pair[0], pair[1])
        for i in range(0, nqubits):
            qc.rx(2 * beta[irep], i)
    qc.measure_all()
    return qc



def create_qaoa_circ2(theta):
    nqubits = 20000
    p = len(theta) // 2  # number of alternating unitaries
    qc = QuantumCircuit(nqubits)
    beta = theta[:p]
    gamma = theta[p:]
    # initial_state
    for i in range(0, nqubits):
        qc.h(i)
    for irep in range(0, p):
        # problem unitary
        for pair in list(edges):
            qc.cx(pair[0], pair[1])
            qc.rz(2 * gamma[irep], qubit=pair[1])
            qc.cx(pair[0], pair[1])
        # mixer unitary
        for i in range(0, nqubits):
            qc.rx(2 * beta[irep], i)
    qc.measure_all()
    return qc













from qiskit.visualization import plot_histogram
backend = Aer.get_backend('qasm_simulator')
backend.shots = 512
time_start = time.time()
qc_res = create_qaoa_circ((1.0, 1.0))
qc_resT = create_qaoa_circ2((1.0, 1.0))
time_end = time.time()
print(time_end - time_start)
time_start = time.time()
qc_res2 = create_qaoa_circ2((1.0, 1.0))
qc_resT2 = create_qaoa_circ2((1.0, 1.0))
time_end = time.time()
print(time_end - time_start)

# print(qc_res)
# counts = backend.run(qc_res, seed_simulator=10).result().get_counts()
#
# a = 0
# print(counts)
# for i in counts:
#     a = counts[i] + a
# print(a)
# # print(counts.keys())
# plot_histogram(counts)
# plt.show()
#



# https://qiskit.org/documentation/tutorials/algorithms/05_qaoa.html