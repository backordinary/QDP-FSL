# https://github.com/Urinx/QuantumComputing/blob/3b2f9719ddca989bda4913db017f7728d47f9297/qiskit_codes/t2_echo.py
# t2_echo.py
import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, register

import Qconfig
register(Qconfig.APItoken, Qconfig.config['url'])

# Define the Quantum and Classical Registers
q = QuantumRegister(1)
c = ClassicalRegister(1)

# Build the circuits
pre = QuantumCircuit(q, c)
pre.h(q)
pre.barrier()
meas_x = QuantumCircuit(q, c)
meas_x.barrier()
meas_x.h(q)
meas_x.measure(q, c)
circuits = []
exp_vector = range(1,51)
for exp_index in exp_vector:
    middle = QuantumCircuit(q, c)
    for i in range(15*exp_index):
        middle.iden(q)
    middle.x(q)
    for i in range(15*exp_index):
        middle.iden(q)
    circuits.append(pre + middle + meas_x)


# Execute the circuits
shots = 1024
job = execute(circuits, 'ibmqx4', shots=shots, max_credits=10)
result = job.result()

# Plot the result
exp_data = []
exp_error = []
for exp_index in exp_vector:
    data = result.get_counts(circuits[exp_index-1])
    try:
        p0 = data['0']/shots
    except KeyError:
        p0 = 0
    exp_data.append(p0)
    exp_error.append(np.sqrt(p0*(1-p0)/shots))

plt.errorbar(exp_vector, exp_data, exp_error)
plt.xlabel('time [31*gate time]')
plt.ylabel('Pr(+)')
plt.grid(True)
plt.show()