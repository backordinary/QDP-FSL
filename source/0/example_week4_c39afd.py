# https://github.com/kaykhos/clifford_project/blob/574255e8e72d4b718ee18d5af8d3c6fc7656d3a5/week4/example_week4.py
# qiskit                0.23.0
# qiskit-aer            0.7.0
# qiskit-aqua           0.8.0
# qiskit-ibmq-provider  0.11.0
# qiskit-ignis          0.5.0
# qiskit-terra          0.16.0
import sys
import circuit_functions
from qiskit.quantum_info import Clifford
from qiskit import Aer
import numpy as np
import qiskit
import random
from docplex.mp.model import Model
from qiskit import QuantumCircuit
from qiskit.aqua.operators import StateFn
import matplotlib.pyplot as plt
qiskit.__qiskit_version__
sys.path.append("/Users/arthur/Desktop/physics/MSci/clifford_project/week4")
# %% generation of a traget state, including randomly select a single qubit gate
qc = QuantumCircuit(4)
qc_initial = qc.copy('qc_initial')
operations_i = ['qc_new.x(2)', 'qc_new.z(1)', 'qc_new.y(3)',
                'qc_new.s(0)']
A = np.array([])
param = random.randint(0, 5)
A = np.append(A, param)
if param == 0:
    qc.id(0)
    operations_i.append('qc_new.id(0)')
elif param == 1:
    qc.h(0)
    operations_i.append('qc_new.h(0)')
elif param == 2:
    qc.s(0)
    operations_i.append('qc_new.s(0)')
elif param == 3:
    qc.x(0)
    operations_i.append('qc_new.x(0)')
elif param == 4:
    qc.y(0)
    operations_i.append('qc_new.y(0)')
elif param == 5:
    qc.z(0)
    operations_i.append('qc_new.z(0)')
param = random.randint(0, 5)
A = np.append(A, param)
if param == 0:
    qc.id(0)
    operations_i.append('qc_new.id(0)')
elif param == 1:
    qc.h(0)
    operations_i.append('qc_new.h(0)')
elif param == 2:
    qc.s(0)
    operations_i.append('qc_new.s(0)')
elif param == 3:
    qc.x(0)
    operations_i.append('qc_new.x(0)')
elif param == 4:
    qc.y(0)
    operations_i.append('qc_new.y(0)')
elif param == 5:
    qc.z(0)
    operations_i.append('qc_new.z(0)')
print(A)


qc.x(2)
qc.z(1)
qc.y(3)
qc.s(0)
qc.cx(0, 1)
# qc.swap(1, 3)
qc.cx(2, 3)


cliff_sf_target = StateFn(qc)

print(qc)
# %% randomly select a single qubit gate and randomly change it to another


# calculate overlap integral


def overlap_modules_square(psi_2):
    psi_1 = cliff_sf_target
    # return (psi_1.adjoint().compose(psi_2).eval().real) * (psi_2.adjoint().compose(psi_1).eval().real)
    return np.real((~psi_1 @ psi_2).eval() * (~psi_2 @ psi_1).eval())


# radom apply GATE_FUNCTIONS
applied_gates = []
gate_list = ['qc_new.id', 'qc_new.x', 'qc_new.y',
             'qc_new.z', 'qc_new.h', 'qc_new.s']


def random_apply_gate(n):
    random_gate = random.choice(gate_list)
    a = '({})'.format(n)
    k = random_gate + a
    exec(k)
    applied_gates.insert(0, k)


# randomly select a gate, and randomly change to a new one
E = np.array([1])
t = np.array([0])
qc_new = qc_initial.copy('qc_new')
operations = operations_i
E1 = 1

for j in range(1000):

    operation = random.choices(operations)  # randomly select gate here
    if operation == operations[0]:
        random_apply_gate(2)
        operations.remove(operations[0])
        operations.insert(0, applied_gates[0])
        exec(operations[1])
        exec(operations[2])
        exec(operations[3])
        exec(operations[4])
        exec(operations[5])
        qc_new.cx(0, 1)
        qc_new.cx(2, 3)
    elif operation == operations[1]:
        random_apply_gate(1)
        operations.remove(operations[1])
        operations.insert(1, applied_gates[0])
        exec(operations[0])
        exec(operations[2])
        exec(operations[3])
        exec(operations[4])
        exec(operations[5])
        qc_new.cx(0, 1)
        qc_new.cx(2, 3)
    elif operation == operations[2]:
        random_apply_gate(3)
        operations.remove(operations[2])
        operations.insert(2, applied_gates[0])
        exec(operations[1])
        exec(operations[0])
        exec(operations[3])
        exec(operations[4])
        exec(operations[5])
        qc_new.cx(0, 1)
        qc_new.cx(2, 3)
    elif operation == operations[4]:
        random_apply_gate(0)
        operations.remove(operations[4])
        operations.insert(4, applied_gates[0])
        exec(operations[1])
        exec(operations[2])
        exec(operations[3])
        exec(operations[0])
        exec(operations[5])
        qc_new.cx(0, 1)
        qc_new.cx(2, 3)
    elif operation == operations[5]:
        random_apply_gate(0)
        operations.remove(operations[5])
        operations.insert(5, applied_gates[0])
        exec(operations[1])
        exec(operations[2])
        exec(operations[3])
        exec(operations[4])
        exec(operations[0])
        qc_new.cx(0, 1)
        qc_new.cx(2, 3)
    else:  # operations[3]
        random_apply_gate(0)
        operations.remove(operations[3])
        operations.insert(3, applied_gates[0])
        exec(operations[1])
        exec(operations[2])
        exec(operations[0])
        exec(operations[4])
        exec(operations[5])
        qc_new.cx(0, 1)
        qc_new.cx(2, 3)

    cliff_sf_output = StateFn(qc_new)
    # E1 = 1
    E2 = 1 - overlap_modules_square(cliff_sf_output)
    # compare between old(E1) and new(E2), and choose whether to accept the change
    if E1 > E2:
        P = 1
    else:  # E1<E2,accept with a probability
        beta = 10
        P = np.exp(beta * (E1 - E2))
    random_prob = random.random()
    if random_prob < P:  # accept the new value
        E = np.append(E, E2)
        E1 = E2
        qc_result = qc_initial
        qc_result = qc_new.copy('qc_result')
        t = np.append(t, j + 1)
    # else:  # not accept the new value
    #     E = np.append(E, E[j - 1])

    qc_new = qc_initial.copy('qc_new_copy')  # initialize the circuit


# %%
# print(E2)
# print(t)
# print(E)
print(qc_result)

plt.figure(figsize=(18, 16), dpi=100)
plt.xlabel('time', fontsize=20)
plt.ylabel('E', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.plot(t, E, marker='o', markerfacecolor='blue',
         markersize=5, color='skyblue')
plt.savefig(fname="Evst.png", figsize=(18, 16), dpi=100)


# %%
def overlap_modules_square(psi_1, psi_2):

    # return (psi_1.adjoint().compose(psi_2).eval().real) * (psi_2.adjoint().compose(psi_1).eval().real)
    return np.real((~psi_1 @ psi_2).eval() * (~psi_2 @ psi_1).eval())


qc1 = circuit_functions.newCircuit(4, 1)
sf_t = StateFn(qc1)
qc1_E = [1]
qc1_t = [0]
qc1_E1 = 1
for a in range(3000):
    qc1 = circuit_functions.updateCircuit(qc1)
    sf_o = StateFn(qc1)
    qc1_E2 = 1 - overlap_modules_square(sf_t, sf_o)
    if qc1_E2 < qc1_E1:
        P = 1
    else:
        beta = 10
        P = np.exp(beta * (qc1_E1 - qc1_E2))
    random_prob = random.random()
    if random_prob < P:
        qc1_E1 = qc1_E2
        qc1_E.append(qc1_E2)
        qc1_t.append(a + 1)

print(qc1)
plt.figure(figsize=(18, 16), dpi=100)
plt.xlabel('time', fontsize=20)
plt.ylabel('E', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.plot(qc1_t, qc1_E, marker='o', markerfacecolor='blue',
         markersize=5, color='skyblue')
plt.savefig(fname="Evst1.png", figsize=(18, 16), dpi=100)
# %%
