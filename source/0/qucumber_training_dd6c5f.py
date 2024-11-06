# https://github.com/Viperr96/Qunatum_things/blob/c52e8e066c7f99b47ec3cec942d58528bf81c6b9/gen/Qucumber_training.py
# import matplotlib.pyplot as plt
import numpy as np
import getpass, time
from math import pi
from h5py import File
from argparse import ArgumentParser
import os, os.path

from qiskit import IBMQ, QuantumCircuit, ClassicalRegister, QuantumRegister, execute, Aer, BasicAer
from qiskit.providers.aer import noise
from qiskit.tools.monitor import job_monitor, backend_monitor, backend_overview
from qiskit.quantum_info.analyzation.average import average_data
from qiskit.providers.ibmq import least_busy
from qiskit.providers.exceptions import JobError, JobTimeoutError
from qiskit.compiler import transpile, assemble

def make_rotation(circuit: QuantumCircuit, registers: QuantumRegister, angles: list):

    """



    Rotates and entangle qubits in the circuit
.





    :param circuit: a QuantumCircuit object comprising two qubits



    :param registers: list of registers involved in the circuit



    :param angles: list of tuples (theta, lambda, phi) with rotation angles



    :return:



    """


    circuit.u3(*(angles[0]), registers[0])



    circuit.u3(*(angles[1]), registers[1])



    circuit.cx(registers[1], registers[0])

IBMQ.save_account('ed1f7070919a8ce0469e69c1cb5b5dc1e114879caada8d0ce25d6e28e91b40c90209146d85eec5118e2584667b02e9662802f0ffaa2494c72375a4906e129fdf', overwrite=True)

provider = IBMQ.load_account()

print("account loaded")
print(provider.backends())
num_genome=1
depth=1
num_qubits=2

theta=np.random.random_sample((num_genome,depth,num_qubits))*pi
phi=np.random.random_sample((num_genome,depth,num_qubits))*2*pi
lada=np.random.random_sample((num_genome,depth,num_qubits))*2*pi


q = QuantumRegister(num_qubits)
c = ClassicalRegister(num_qubits)
singlet = QuantumCircuit(q, c)
igenome = 0
for idepth in range(depth):
    lada[igenome, idepth, 0] = 0
    lada[igenome, idepth, 1] = 0

    make_rotation(singlet, q, [(theta[igenome, idepth, 0], phi[igenome, idepth, 0], lada[igenome, idepth, 0]),
                               (theta[igenome, idepth, 1], phi[igenome, idepth, 1], lada[igenome, idepth, 1])])

measureZZ = QuantumCircuit(q, c)
measureZZ.measure(q[0], c[0])
measureZZ.measure(q[1], c[1])
singletZZ = singlet+measureZZ

backend = provider.get_backend('ibmq_16_melbourne')
device_shots = 2048

job = execute(singletZZ, backend=backend, shots=device_shots)
job_monitor(job)
result = job.result()
res = result.get_counts(singletZZ)
print(res)
# no_error = True
# while no_error:
#     try:
#         # job = execute(circuits, BasicAer.get_backend('qasm_simulator'), shots=device_shots)
#         # job = do_job_on_simulator(backend, circuits)
#
#         job = execute(singletZZ, backend=backend, shots=device_shots)
#         job_monitor(job)
#         result = job.result()
#         no_error = False
#     except JobError:
#         print(JobError)
#         no_error = True