# https://github.com/takehuge/Qalgorithm/blob/3f26c624ff38e03e915b3949d583cb747d73aca3/QisKit/oldxample.py
import Qconfig
# print(Qconfig.APItoken, Qconfig.config['url'])

import getpass
import time
from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
from qiskit import available_backends, execute, register
from qiskit.backends.ibmq import least_busy

# import basic plot tools
from qiskit.tools.visualization import plot_histogram, circuit_drawer

# APItoken = getpass.getpass('Please input your token and hit enter: ')
# qx_config = {
#     "APItoken": APItoken,
#     "url": "https://quantumexperience.ng.bluemix.net/api"}

try:
    # qx_config['APItoken'], qx_config['url']
    register(Qconfig.APItoken, Qconfig.config['url'])

    print('\nYou have access to great power!')
    print(available_backends({'local': False, 'simulator': False}))
except:
    print('Something went wrong.\nDid you enter a correct token?')

backend = least_busy(available_backends({'simulator': False, 'local': False}))
print("The least busy backend is " + backend)

