# https://github.com/NathanLG/quantum-cuts/blob/931b1a69cee0b685734b75a0e6f921fefa425bf5/aqft_experiment.py
from qiskit import Aer, IBMQ, execute, QuantumCircuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.converters import circuit_to_dag
from qiskit.visualization import dag_drawer, circuit_drawer
import numpy as np

from utils.helper_fun import *

from all_cutter import *

import time
from datetime import datetime
import matplotlib.pyplot as plt

sizes = []
time_wises = []
space_wises = []

counter = 0
for i in range(12,16,2):
    testc = generate_circ(i, "aqft")
    print("Cutting circuits of size " + str(i))
    sizes.append(i)

    millis = int(round(time.time() * 1000))
    for a in range(20):
        ey = mixed_cut(testc, i,  4 + counter, [(5 + counter,6,"time"), (6 + counter,4,"time"), (7 + counter,2,"time"), (4 + counter,8,"time")], 8 + counter, 8 + counter)
    t_time = int(round(time.time() * 1000)) - millis
    print(t_time)
    time_wises.append(t_time)

    testc = generate_circ(i, "aqft")
    millis = int(round(time.time() * 1000))
    for a in range(20):
        ey = mixed_cut(testc, i,  5 + counter, [(5 + counter,6,"time"), (6 + counter,4,"time"), (7 + counter,2,"time"), (4 + counter,8,"space")], 8 + counter, 7 + counter)
    s_time = int(round(time.time() * 1000)) - millis
    print(s_time)
    space_wises.append(s_time)

    counter += 1

c = 0
for i in range(16,29,2):
    print("Cutting circuits of size " + str(i))
    sizes.append(i)
    
    testc = generate_circ(i, "aqft")
    millis = int(round(time.time() * 1000))
    for a in range(20):
        ey = mixed_cut(testc, i,  5 + c, [(6 + c,8,"time"), (7 + c,6,"time"), (8 + c,4,"time"), (9 + c,2,"time"), (5 + c,10,"time")], 10 + c, 11 + c)
    t_time = int(round(time.time() * 1000)) - millis
    print(t_time)
    time_wises.append(t_time)

    testc = generate_circ(i, "aqft")
    millis = int(round(time.time() * 1000))
    for a in range(20):
        ey = mixed_cut(testc, i,  6 + c, [(6 + c,8,"time"), (7 + c,6,"time"), (8 + c,4,"time"), (9 + c,2,"time"), (5 + c,10,"space")], 10 + c, 10 + c)
    s_time = int(round(time.time() * 1000)) - millis
    print(s_time)
    space_wises.append(s_time)
    
    c += 1

time_wises = np.array(time_wises)
space_wises = np.array(space_wises)

plt.plot(sizes, (time_wises - space_wises))
plt.yscale('log')
plt.savefig("time_mix_difflog_" + str(datetime.today()) + ".jpg")
plt.show()

plt.plot(sizes, time_wises - space_wises)
plt.savefig("time_mix_diff_" + str(datetime.today()) + ".jpg")
plt.show()

plt.plot(sizes, time_wises, color='blue')
plt.plot(sizes, space_wises, color='red')
plt.yscale('log')
plt.savefig("time_mix_" + str(datetime.today()) + ".jpg")
plt.show()

plt.plot(sizes, (time_wises - space_wises)/time_wises)
plt.savefig("time_mix_diff_prop_" + str(datetime.today()) + ".jpg")
plt.show()
