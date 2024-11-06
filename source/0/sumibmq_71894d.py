# https://github.com/AlbertoVari/Qadder/blob/17aa4b15dfe24f25971f22ef4017c16de29509e6/sumIBMQ.py
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
# exit(
# Simulation)
# backend = provider.get_backend('ibmq_qasm_simulator')
backend = provider.get_backend('ibmq_lima')
status = backend.status()
is_operational = status.operational
jobs_in_queue = status.pending_jobs
print(is_operational,jobs_in_queue)
q0 = input ("Insert Q0 -> ")
q1 = input ("Insert Q1 -> ")
q0 = int(q0)
q1 = int(q1)
qreg_q = QuantumRegister(4, 'q')
creg_c = ClassicalRegister(4, 'c')
circuit = QuantumCircuit(qreg_q, creg_c)
if q0 == 1 :
    circuit.x(qreg_q[0])
if q1 == 1 :
    circuit.x(qreg_q[1])
circuit.ccx(qreg_q[0], qreg_q[1], qreg_q[3])
circuit.cx(qreg_q[0], qreg_q[1])
circuit.ccx(qreg_q[1], qreg_q[2], qreg_q[3])
circuit.cx(qreg_q[1], qreg_q[2])
circuit.cx(qreg_q[0], qreg_q[1])
circuit.measure(qreg_q[0], creg_c[0])
circuit.measure(qreg_q[1], creg_c[1])
circuit.measure(qreg_q[2], creg_c[2])
circuit.measure(qreg_q[3], creg_c[3])
print(circuit)
num_shots = 1000
job = execute(circuit, backend, shots=num_shots)
result = job.result()
counts = result.get_counts(circuit)
# print("Key: ",counts.keys(),"values: ",counts.values())
all_values=[''] * 16
all_counts=[0] * 16
sort_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
print("Res  Counts")
for i in sort_counts:
        print(i[0], i[1])
#counts is a dictionary -  all_values is a list
cnts = counts.values()
values  = counts.keys()
all_values = list(values)
all_counts = list(cnts)
value_idx =  all_counts.index(max(all_counts))
value_n = int(value_idx)
mx_value = all_values[value_n]
print(" ")
print("Result : CARRY SUM Q1 Q0 ")
print("         ", mx_value)
