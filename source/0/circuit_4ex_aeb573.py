# https://github.com/hoang-ho/Quantum_Computing/blob/d494582589439ef0bf43ffcde27b711e2cfba2f2/quantum_gates/circuit_4ex.py
import getpass
import time
from qiskit import ClassicalRegister, QuantumRegister, QuantumProgram, QuantumCircuit
from qiskit import available_backends, execute, register, least_busy, register

# import basic plot tools
from qiskit.tools.visualization import plot_histogram, matplotlib_circuit_drawer

import matplotlib.pyplot as plt
import numpy as np
from math import pi
import math


# APItoken = getpass.getpass('Please input your token and hit enter: ')
# qx_config = {
#     "APItoken": APItoken,
#     "url": "https://quantumexperience.ng.bluemix.net/api"}
# print('Qconfig.py not found in qiskit-tutorial directory; Qconfig loaded using user input.')

APItoken = getpass.getpass('Please input your token and hit enter: ')
qx_config = {
    "APItoken": APItoken,
    "url": "https://quantumexperience.ng.bluemix.net/api"}

try:
    register(qx_config['APItoken'], qx_config['url'])

    print('\nYou have access to great power!')
    print(available_backends({'local': False, 'simulator': False}))
except:
    print('Something went wrong.\nDid you enter a correct token?')

# backend = least_busy(available_backends({'simulator': False, 'local': False}))
# print("The least busy backend is " + backend)

'''
using load_random from Iris.py to obtain some training and test examples
for example:
sample_train
array([[-0.99407732, -0.10867513,  0.        ],
       [-0.30654501,  0.95185616,  0.        ],
       [-0.92072409, -0.39021423,  1.        ],
       [ 0.88418237, -0.46714188,  1.        ]])


sample_test
array([[-0.8448179 ,  0.53505394,  0.        ],
       [-0.36817031, -0.92975837,  1.        ]])
'''

q = QuantumRegister(5)
c = ClassicalRegister(5)

qc = QuantumCircuit(q, c)
# Step A: prepare the index and ancilla qubits in superposition
qc.h(q[0])
qc.h(q[1])
qc.h(q[2])

# Step B: entangle the test example with the ground state of the ancilla
# qc.cry(2 * math.acos(-0.8448179), q[0], q[3])
# qc.cry(2 * (2 * pi - math.acos(-0.36817031)), q[0], q[3])
qc.cry(2 * math.acos(-0.36817031), q[0], q[3])
qc.cz(q[0], q[3])
# qc.crz(math.pi, q[0], q[3])
qc.x(q[0])

# Step C: entangle the first trainning example with the excited state of the ancilla and the ground states of the two index qubits
# qc.cccry(2 * (2 * pi - math.acos(-0.99407732)), q[0], q[1], q[2], q[3])
qc.cccry(2 * math.acos(-0.99407732), q[0], q[1], q[2], q[3])
# qc.cccrz(math.pi, q[0], q[1], q[2], q[3])
qc.cccz(q[0], q[1], q[2], q[3])
qc.x(q[1])
qc.x(q[2])

# Step D: entangle the second trainning example with the excited state of the ancilla and the LSB and the ground state of the MSB of the index qubits
qc.cccry(2 * math.acos(-0.30654501), q[0], q[1], q[2], q[3])
qc.cx(q[1], q[2])
qc.swap(q[1], q[2])

# Step E: entangle the third trainning example with the excited state of the ancilla and the MSB and the ground state of the LSB of the index qubits
# qc.cccry(2 * (2 * pi - math.acos(-0.92072409)), q[0], q[1], q[2], q[3])
qc.cccry(2 * math.acos(-0.92072409), q[0], q[1], q[2], q[3])
# qc.cccrz(math.pi, q[0], q[1], q[2], q[3])
qc.cccz(q[0], q[1], q[2], q[3])
qc.cx(q[1], q[2])

# Step F: entangle the fourth trainning example with the excited state of the ancilla and the index qubits
# qc.cccry(2 * (2 * pi - math.acos(0.88418237)), q[0], q[1], q[2], q[3])
qc.cccry(2 * math.acos(0.88418237), q[0], q[1], q[2], q[3])
# qc.cccrz(math.pi, q[0], q[1], q[2], q[3])
qc.cccz(q[0], q[1], q[2], q[3])

# Step G: Swap the data and the register for the LSB of the index qubit for the class qubit
qc.swap(q[2], q[4])
qc.cx(q[1], q[2])

qc.h(q[0])

qc.barrier(q)
qc.measure(q[0], c[4])
qc.measure(q[1], c[3])
qc.measure(q[2], c[2])
qc.measure(q[3], c[1])
qc.measure(q[4], c[0])


# job = execute(qc, backend='local_qasm_simulator', shots=8192)
# print(job.result().get_counts(qc))
# plot_histogram(job.result().get_counts(qc))

job_exp = execute(qc, backend="ibmqx5", shots=8192, max_credits=3)

lapse = 0
interval = 60
while not job_exp.done:
    print('Status @ {} seconds'.format(interval * lapse))
    print(job_exp.status)
    time.sleep(interval)
    lapse += 1
print(job_exp.status)

sim_result = job_exp.result()

plot_histogram(sim_result.get_counts(qc))
