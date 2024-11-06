# https://github.com/jcylim/QiskitProject/blob/92131a069db06122ee5196bd0fd3d06f8d54bfff/qiskit/quantum%20information%20science/entanglementExtended.py
# Imports
import matplotlib.pyplot as plt
#%matplotlib inline
import numpy as np

from qiskit import QuantumProgram, QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import available_backends, get_backend, execute, register, least_busy
from qiskit.tools.visualization import plot_histogram, circuit_drawer

# Connecting to the IBM Quantum Experience
qx_config = {
  "APItoken": 'dcdb2d9414a625c1f57373c544add3711c78c3d7faf39397fe2c41887110e8b59caf81bcb2bc32714d936da41a261fea510f96df379afcbdfa9df6cc6bfe3829',
    "url": 'https://quantumexperience.ng.bluemix.net/api'
}
register(qx_config['APItoken'], qx_config['url'])
device_shots = 1024
device_name = least_busy(available_backends({'simulator': False, 'local': False}))
device = get_backend(device_name)
device_coupling = device.configuration['coupling_map']
print("the best backend is " + device_name + " with coupling " + str(device_coupling))
# Creating registers
q = QuantumRegister(2)
c = ClassicalRegister(2)

# quantum circuit to make an entangled bell state
bell = QuantumCircuit(q, c)
bell.h(q[0])
bell.cx(q[0], q[1])

# quantum circuit to measure q in the standard basis
measureZZ = QuantumCircuit(q, c)
measureZZ.measure(q[0], c[0])
measureZZ.measure(q[1], c[1])
bellZZ = bell+measureZZ

# quantum circuit to measure q in the superposition basis
measureXX = QuantumCircuit(q, c)
measureXX.h(q[0])
measureXX.h(q[1])
measureXX.measure(q[0], c[0])
measureXX.measure(q[1], c[1])
bellXX = bell+measureXX

# quantum circuit to measure ZX
measureZX = QuantumCircuit(q, c)
measureZX.h(q[0])
measureZX.measure(q[0], c[0])
measureZX.measure(q[1], c[1])
bellZX = bell+measureZX

# quantum circuit to measure XZ
measureXZ = QuantumCircuit(q, c)
measureXZ.h(q[1])
measureXZ.measure(q[0], c[0])
measureXZ.measure(q[1], c[1])
bellXZ = bell+measureXZ

circuits = [bellZZ,bellXX,bellZX,bellXZ]
job = execute(circuits, backend=device_name, coupling_map=device_coupling, shots=device_shots)
result = job.result()
observable_first ={'00': 1, '01': -1, '10': 1, '11': -1}
observable_second ={'00': 1, '01': 1, '10': -1, '11': -1}
observable_correlated ={'00': 1, '01': -1, '10': -1, '11': 1}
print('IZ = ' + str(result.average_data(bellZZ,observable_first)))
print('ZI = ' + str(result.average_data(bellZZ,observable_second)))
print('ZZ = ' + str(result.average_data(bellZZ,observable_correlated)))

print('IX = ' + str(result.average_data(bellXX,observable_first)))
print('XI = ' + str(result.average_data(bellXX,observable_second)))
print('XX = ' + str(result.average_data(bellXX,observable_correlated)))

print('ZX = ' + str(result.average_data(bellZX,observable_correlated)))
print('XZ = ' + str(result.average_data(bellXZ,observable_correlated)))