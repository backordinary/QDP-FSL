# https://github.com/jcylim/QiskitProject/blob/92131a069db06122ee5196bd0fd3d06f8d54bfff/qiskit/quantum%20information%20science/teleportation.py
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
tq = QuantumRegister(3)
tc0 = ClassicalRegister(1)
tc1 = ClassicalRegister(1)
tc2 = ClassicalRegister(1)

# Quantum circuit to make the shared entangled state
teleport = QuantumCircuit(tq, tc0,tc1,tc2)
teleport.h(tq[1])
teleport.cx(tq[1], tq[2])
teleport.ry(np.pi/4,tq[0])
teleport.cx(tq[0], tq[1])
teleport.h(tq[0])
teleport.barrier()
teleport.measure(tq[0], tc0[0])
teleport.measure(tq[1], tc1[0])
teleport.z(tq[2]).c_if(tc0, 1)
teleport.x(tq[2]).c_if(tc1, 1)
teleport.measure(tq[2], tc2[0])
teleport_job = execute(teleport, 'local_qasm_simulator') # note that this circuit doesn't run on a real device
teleport_result = teleport_job.result()
data = teleport_result.get_counts(teleport)
print(data)
'''alice = {}
alice['00'] = data['0 0 0'] + data['1 0 0']
alice['10'] = data['0 1 0'] + data['1 1 0']
alice['01'] = data['0 0 1'] + data['1 0 1']
alice['11'] = data['0 1 1'] + data['1 1 1']
plot_histogram(alice)'''
bob = {}
bob['0'] = data['0 0 0'] + data['0 1 0'] +  data['0 0 1'] + data['0 1 1']
bob['1'] = data['1 0 0'] + data['1 1 0'] +  data['1 0 1'] + data['1 1 1']
plot_histogram(bob)