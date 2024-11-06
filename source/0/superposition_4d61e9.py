# https://github.com/jcylim/QiskitProject/blob/92131a069db06122ee5196bd0fd3d06f8d54bfff/qiskit/quantum%20information%20science/superposition.py
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
'''register(qx_config['APItoken'], qx_config['url'])
backend = least_busy(available_backends({'simulator': False, 'local': False}))
print("the best backend is " + backend)
'''
# Creating registers
qr = QuantumRegister(1)
cr = ClassicalRegister(1)

# Quantum circuit ground
qc_ground = QuantumCircuit(qr, cr)
qc_ground.measure(qr[0], cr[0])

# Quantum circuit excited
qc_excited = QuantumCircuit(qr, cr)
qc_excited.x(qr)
qc_excited.measure(qr[0], cr[0])

# Quantum circuit superposition
qc_superposition = QuantumCircuit(qr, cr)
qc_superposition.h(qr)
qc_superposition.barrier()
qc_superposition.h(qr)
qc_superposition.measure(qr[0], cr[0])

circuits = [qc_ground, qc_excited, qc_superposition]
job = execute(circuits, backend)
result = job.result()
plot_histogram(result.get_counts(qc_superposition))