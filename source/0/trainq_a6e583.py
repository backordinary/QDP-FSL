# https://github.com/AlbertoVari/SolidQML/blob/c8b520ce050d55e198185985c44c2b887e22a6f8/trainQ.py

# Importing standard Qiskit libraries and configuring account
from qiskit import QuantumCircuit, execute, Aer, IBMQ, QuantumRegister, ClassicalRegister
from qiskit.compiler import transpile, assemble
from qiskit.tools.jupyter import *
from qiskit.visualization import *
from numpy import pi
import json

# Loading your IBM Q account(s)
IBMQ.load_account()
provider = IBMQ.load_account()
print(provider.backends())
# exit()

#backend = provider.get_backend('ibmq_manila')
backend = provider.get_backend('ibmq_qasm_simulator')
status = backend.status()
is_operational = status.operational
jobs_in_queue = status.pending_jobs
print(is_operational,jobs_in_queue)

qreg_q = QuantumRegister(1, 'q')
creg_c = ClassicalRegister(1, 'c')
circuit = QuantumCircuit(qreg_q, creg_c)
circuit.h(qreg_q[0])
circuit.barrier(qreg_q[0])
circuit.ry(pi, qreg_q[0])
circuit.barrier(qreg_q[0])
circuit.measure(qreg_q[0], creg_c[0])
print(circuit)

num_shots = 1000
job = execute(circuit, backend, shots=num_shots)
result = job.result()
counts = result.get_counts(circuit)
print("Result : ",counts)
hybrid = counts["1"]/num_shots
print("Stato 1 Prob",hybrid)
 


