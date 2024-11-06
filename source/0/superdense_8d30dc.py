# https://github.com/jcylim/QiskitProject/blob/92131a069db06122ee5196bd0fd3d06f8d54bfff/qiskit/quantum%20information%20science/superdense.py
# useful additional packages
import matplotlib.pyplot as plt
#%matplotlib inline
import numpy as np
from pprint import pprint

# importing QISKit
from qiskit import QuantumProgram, QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import available_backends, execute, register, least_busy

# import basic plot tools
from qiskit.tools.visualization import plot_histogram, circuit_drawer

qx_config = {
  "APItoken": 'dcdb2d9414a625c1f57373c544add3711c78c3d7faf39397fe2c41887110e8b59caf81bcb2bc32714d936da41a261fea510f96df379afcbdfa9df6cc6bfe3829',
    "url": 'https://quantumexperience.ng.bluemix.net/api'
}

backend = 'local_qasm_simulator' # run on local simulator by default
# register(qx_config['APItoken'], qx_config['url'])
# backend = least_busy(available_backends({'simulator': False, 'local': False}))
# print("the best backend is " + backend)

# Creating registers
sdq = QuantumRegister(2)
sdc = ClassicalRegister(2)

# Quantum circuit to make the shared entangled state
superdense = QuantumCircuit(sdq, sdc)
superdense.h(sdq[0])
superdense.cx(sdq[0], sdq[1])

# For 00, do nothing

# For 01, apply $X$
#shared.x(q[0])

# For 01, apply $Z$
#shared.z(q[0])

# For 11, apply $XZ$
superdense.z(sdq[0])
superdense.x(sdq[0])
superdense.barrier()

superdense.cx(sdq[0], sdq[1])
superdense.h(sdq[0])
superdense.measure(sdq[0], sdc[0])
superdense.measure(sdq[1], sdc[1])
superdense_job = execute(superdense, backend)
superdense_result = superdense_job.result()

plot_histogram(superdense_result.get_counts(superdense))