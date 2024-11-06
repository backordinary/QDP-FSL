# https://github.com/tgag17/BTechProject/blob/6e5822f2b73725db3e3f990a847446ed81997089/QSim/basic_exp/tests.py
from qiskit import QuantumRegister, ClassicalRegister
from qiskit import QuantumCircuit, execute, BasicAer
import numpy as np
import time

#Options & Noise goes here - Don't change options variable name & block
options = { 

"rotation_error": {'rx':[1.0, 0.0], 'ry':[1.0, 0.0], 'rz':[1.0,0.0]},
"tsp_model_error": [1.0, 0.0],
"thermal_factor": 1.0,
"decoherence_factor": 1.0,
"depolarization_factor": 1.0,
"bell_depolarization_factor": 1.0,
"decay_factor": 1.0,
}

backend = BasicAer.get_backend('dm_simulator')

## 1I gate
#######################################

qc = QuantumCircuit()
q = QuantumRegister(1, 'q')
c1 = ClassicalRegister(1, 'c1')

qc.add_register(q)
qc.add_register(c1)

qc.i(q[0])
qc.measure(q[0], c1[0])

tic = time.time()
job = execute(qc, backend=backend, **options)
toc = time.time()
job_result = job.result()

t = toc - tic
print(t)

## 1X gate
#######################################

qc = QuantumCircuit()
q = QuantumRegister(1, 'q')
c1 = ClassicalRegister(1, 'c1')

qc.add_register(q)
qc.add_register(c1)

qc.x(q[0])
qc.measure(q[0], c1[0])

tic = time.time()
job = execute(qc, backend=backend, **options)
toc = time.time()
job_result = job.result()

t = toc - tic
print(t)

## 5X gate - inline
#######################################

qc = QuantumCircuit()
q = QuantumRegister(1, 'q')
c1 = ClassicalRegister(1, 'c1')

qc.add_register(q)
qc.add_register(c1)
qc.x(q[0])
qc.x(q[0])
qc.x(q[0])
qc.x(q[0])
qc.x(q[0])
qc.measure(q[0], c1[0])

tic = time.time()
job = execute(qc, backend=backend, **options)
toc = time.time()
job_result = job.result()

t = toc - tic
print(t)

## 5X gate - parallel
#######################################

qc = QuantumCircuit()
q = QuantumRegister(5, 'q')
c1 = ClassicalRegister(5, 'c1')

qc.add_register(q)
qc.add_register(c1)
qc.x(q[0])
qc.x(q[1])
qc.x(q[2])
qc.x(q[3])
qc.x(q[4])
qc.measure(q[0:5], c1[0:5])

tic = time.time()
job = execute(qc, backend=backend, **options)
toc = time.time()
job_result = job.result()

t = toc - tic
print(t)

## 7X gate - parallel
#######################################

qc = QuantumCircuit()
q = QuantumRegister(7, 'q')
c1 = ClassicalRegister(7, 'c1')

qc.add_register(q)
qc.add_register(c1)
qc.x(q[0])
qc.x(q[1])
qc.x(q[2])
qc.x(q[3])
qc.x(q[4])
qc.x(q[5])
qc.x(q[6])
qc.measure(q[0:7], c1[0:7])

tic = time.time()
job = execute(qc, backend=backend, **options)
toc = time.time()
job_result = job.result()

t = toc - tic
print(t)

## 10X gate - parallel
#######################################

qc = QuantumCircuit()
q = QuantumRegister(10, 'q')
c1 = ClassicalRegister(10, 'c1')

qc.add_register(q)
qc.add_register(c1)
qc.x(q[0])
qc.x(q[1])
qc.x(q[2])
qc.x(q[3])
qc.x(q[4])
qc.x(q[5])
qc.x(q[6])
qc.x(q[7])
qc.x(q[8])
qc.x(q[9])
qc.measure(q[0:10], c1[0:10])

tic = time.time()
job = execute(qc, backend=backend, **options)
toc = time.time()
job_result = job.result()

t = toc - tic
print(t)