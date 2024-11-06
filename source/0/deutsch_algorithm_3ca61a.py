# https://github.com/Thakkar-meet/Quantum-Computing/blob/1b921565abe4993cd8645531e1d7030f1b19f3bb/Deutsch_Algorithm.py
from qiskit import QuantumCircuit, QuantumRegister, assemble, Aer
import matplotlib.pyplot as plt
import random
from qiskit.visualization import plot_histogram
from qiskit.compiler import transpile


def draw(qc):
    qc.draw(output='mpl')
    plt.show()


def random_key(p):
    key = ""
    for i in range(p):
        tmp = str(random.randint(0,1))
        key += tmp
    return key


def oracle(case, n):
    oracle_qc = QuantumCircuit(n+1)
    if case == "balanced":
        b = random_key(n)
        for i in range(len(b)):
            if b[i] == "1":
                oracle_qc.x(i)

        for i in range(n):
            oracle_qc.cx(i, n)

        for i in range(len(b)):
            if b[i] == "1":
                oracle_qc.x(i)

    if case == "constant":
        output = random.randint(0,1)
        if output == 1:
            oracle_qc.x(n)
    draw(oracle_qc)
    oracle_gate = oracle_qc.to_gate()
    oracle_gate.name = "Oracle"
    return oracle_gate


def algorithm(oracle, n):
    qc = QuantumCircuit(n+1,n)
    for i in range(n):
        qc.h(i)
    qc.x(n)
    qc.h(n)
    qc.append(oracle, range(n+1))
    for i in range(n):
        qc.h(i)

    for i in range(n):
        qc.measure(i,i)

    return qc


n = 5
oracle = oracle("constant", n)
dj_circuit = algorithm(oracle, n)
draw(dj_circuit)

aer_sim = Aer.get_backend("aer_simulator")
transpiled_dj_circuit = transpile(dj_circuit, aer_sim)
qobj = assemble(transpiled_dj_circuit)
results = aer_sim.run(qobj).result()
answer = results.get_counts()
plot_histogram(answer)
plt.show()