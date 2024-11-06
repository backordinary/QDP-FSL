# https://github.com/lukasszz/qiskit-exp/blob/ce14d53735870e7b6ace352629eb4049e9cd6740/stack_q0_grover.py
from qiskit import *
from qiskit.tools.visualization import *


def ccz(qci, q0, q1, q2):
    qci.h(q2)
    qci.ccx(q0, q1, q2)
    qci.h(q2)


def grover(qci, q0, q1, q2):
    ccz(qci, q0, q1, q2)
    for i in [q0, q1, q2]:
        qci.h(i)
        qci.x(i)
    ccz(qci, q0, q1, q2)
    for i in [q0, q1, q2]:
        qci.x(i)
        qci.h(i)



bn = 3
q = QuantumRegister(bn)
c = ClassicalRegister(bn)
qc = QuantumCircuit(q, c)
for i in range(bn):
    qc.h(q[i])
grover(qc, q[0], q[1], q[2])
for i in range(bn):
    qc.measure(q[bn - i - 1], c[i])
r = execute(qc, "local_qasm_simulator").result()
rc = r.get_counts()
print(rc)
plot_histogram(rc)

