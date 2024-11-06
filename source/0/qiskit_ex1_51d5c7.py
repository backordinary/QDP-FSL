# https://github.com/hoang-ho/Quantum_Computing/blob/d494582589439ef0bf43ffcde27b711e2cfba2f2/quantum_gates/Qiskit_ex1.py
import getpass
import time
from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
from qiskit import available_backends, execute, register, least_busy

# import basic plot tools
from qiskit.tools.visualization import plot_histogram, matplotlib_circuit_drawer

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

backend = least_busy(available_backends({'simulator': False, 'local': False}))
print("The least busy backend is " + backend)

q = QuantumRegister(2)
c = ClassicalRegister(2)
qc = QuantumCircuit(q, c)
qc.h(q[0])
qc.cx(q[0], q[1])
qc.measure(q, c)
job_exp = execute(qc, backend=backend, shots=1024, max_credits=3)

plot_histogram(job_exp.result().get_counts(qc))

matplotlib_circuit_drawer(qc)
