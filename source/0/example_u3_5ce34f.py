# https://github.com/oimichiu/quantumGateModel/blob/fa1cb5ed751edbebe512ee299d5484949c856340/IBMQX/qiskit-tutorials/coduriCareNUcompileaza/tutoriale-QISKit/01-getting_started/example_u3.py
# example_u3.py
import sys
import os
import time

# import qiskit from qiskit-sdk-py folder
try:
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../', 'qiskit-sdk-py'))
    import Qconfig
    qx_config = {
        "APItoken": Qconfig.APItoken,
        "url": Qconfig.config['url']}
except:
    qx_config = {
        "APItoken":"da2a4002660558a35103a600bcbda7fe438cea629a6be98969ea5e367c091b6815e624bd86b6207121bd97fef79c22033318a4402eeafcbd04b021fd80f5a195",
        "url":"https://quantumexperience.ng.bluemix.net/api"
    }

import numpy as np 
import matplotlib.pyplot as plt 

from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit, execute

# Define the Classical and Quantum Registers
c = ClassicalRegister(1)
q = QuantumRegister(1)

# Build the circuits
circuits = []
middle = QuantumCircuit(q, c)
meas = QuantumCircuit(q, c)
meas.barrier()
meas.measure(q, c)
exp_vector = range(0, 50)
exp_theta = []
theta = 0.0
for exp_index in exp_vector:
    delta_theta = 2*np.pi/len(exp_vector)
    theta = theta + delta_theta
    exp_theta.append(theta)
    middle.u3(delta_theta, 0, 0, q)
    circuits.append(middle + meas)

# Execute the circuits
shots = 4096
job = execute(circuits, backend = 'local_qasm_simulator', shots=shots, seed = 8)
result = job.result()

# Plot result
exp_data = []
exp_error = []
for exp_index in exp_vector:
    data = result.get_counts(circuits[exp_index])
    try:
        p0 = data['0']/shots
    except KeyError:
        p0 = 0
    exp_data.append(p0)
    exp_error.append(np.sqrt(p0*(1-p0)/shots))

plt.errorbar(exp_theta, exp_data, exp_error)
plt.xlabel('theta')
plt.ylabel('Pr(0)')
plt.grid(True)
plt.show()