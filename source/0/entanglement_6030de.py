# https://github.com/clawyq/qiskit/blob/97735cde61826029d236098342b1503e6352b60f/entanglement.py
import qiskit
from qiskit import *
from qiskit.tools.monitor import job_monitor

import os
from os.path import join, dirname
from dotenv import load_dotenv

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

#circuit with two qubit and two classical registers
qc = QuantumCircuit(2,2)

# qc.h(0)
# qc.x(1)
'''
together, this operation is represented as X|q1> __tensor_product__ H|q0> = (X __tensor_product__H)|q1q0>
whihc is simply 'notting' the qubits and then applying the hadamard operator 1/sqrt(2) [[1, 1], [1, -1]] = [[0, H], [H, 0]]
'''
qc.h(0)
# cnot gate(control qubit, target qubit)
#  The CNOT gate flips the second qubit (the target qubit) iff the first qubit (the control qubit) is |1>.
qc.cx(0,1) 

qc.measure([0,1],[0,1])
qc.draw('mpl')

IBMQ.save_account(os.environ.get("IBM_API_TOKEN"))
provider = IBMQ.load_account()

device = provider.get_backend('ibmq_16_melbourne')     #we use ibmq_16_melbourne quantum device 
job = execute(qc, backend=device) #we pass our circuit and backend as usual 
job_monitor(job)    #to see our status in queue

result = job.results()

counts = result.get_counts(qc)

print (counts)
