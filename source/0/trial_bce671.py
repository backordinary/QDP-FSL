# https://github.com/hoang-ho/Quantum_Computing/blob/d494582589439ef0bf43ffcde27b711e2cfba2f2/quantum_gates/trial.py
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

# APItoken = getpass.getpass('Please input your token and hit enter: ')
# qx_config = {
#     "APItoken": APItoken,
#     "url": "https://quantumexperience.ng.bluemix.net/api"}
# print('Qconfig.py not found in qiskit-tutorial directory; Qconfig loaded using user input.')

q = QuantumRegister(4)
c = ClassicalRegister(4)
qc = QuantumCircuit(q, c)

'''
using load_random from Iris.py to obtain some training and test examples
for example:
sample_train
array([[-0.65752458,  0.75343309,  0.        ],
       [-0.6027035 ,  0.79796522,  0.     ,  1.        ],
       [ 0.92790088, -0.37282697,  1.        ]])


sample_test
array([[-0.26461538,  0.96435403,  0.        ],
       [-0.23192983, -0.97273252,  1.        ]])
'''


# Step A: prepare the index qubit and ancilla qubit in superposition
qc.h(q[0])
qc.h(q[1])

# Step B: entangle the test data with the ground state of the ancilla
qc.cry(2 * math.acos(-0.26461538), q[0], q[2])
# qc.cry(2 * (2 * pi - math.acos(-0.23192983)), q[0], q[2])
# qc.cry(2 * math.acos(-0.23192983), q[0], q[2])
# qc.crz(math.pi, q[0], q[2])
# qc.cz(q[0], q[2])
qc.x(q[0])

# Step C: entangle the training data with the excited state of the ancilla and the ground state of the index qubit
qc.ccry(2 * math.acos(-0.65752458), q[0], q[1], q[2])
qc.x(q[1])

# Step D: entangle the training data with the excited state of the ancilla and of the index qubit
# qc.ccry(2 * (2 * pi - math.acos(0.92790088)), q[0], q[1], q[2])
qc.ccry(2 * math.acos(0.92790088), q[0], q[1], q[2])
# qc.ccrz(math.pi, q[0], q[1], q[2])
qc.ccz(q[0], q[1], q[2])

# Step E: Swap the data and the class qubits (due to the topology of IBM 5 QX)
# and the class qubit is flipped given the index qubit is 1
qc.swap(q[2], q[3])
qc.cx(q[1], q[2])

# Step F: Hadamard gate on the ancilla qubit and measurement
qc.h(q[0])

# Measurement
qc.barrier(q)
qc.measure(q[0], c[3])
qc.measure(q[1], c[2])
qc.measure(q[2], c[1])
qc.measure(q[3], c[0])

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

# Show the results
print("simulation: ", sim_result)
plot_histogram(sim_result.get_counts(qc))

# job = execute(qc, backend='local_qasm_simulator', shots=8192)
# print(job.result().get_counts(qc))
# plot_histogram(job.result().get_counts(qc))
