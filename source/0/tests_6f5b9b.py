# https://github.com/tgag17/BTechProject/blob/d972d2aa8909571677ff034573170614d6dbd75c/Qiskit/basic_exp/tests.py
from qiskit import QuantumCircuit, Aer
import time

sim = Aer.get_backend('aer_simulator')

## 1I gate
####################################

qc = QuantumCircuit(1)
qc.i(0)
qc.measure_all()

tic = time.time()
result = sim.run(qc).result()
toc = time.time()

t = toc - tic
print(t)

## 1X gate
####################################

qc = QuantumCircuit(1)
qc.x(0)
qc.measure_all()

tic = time.time()
result = sim.run(qc).result()
toc = time.time()

t = toc - tic
print(t)

## 5X gate - inline 
####################################

qc = QuantumCircuit(1)
qc.x(0)
qc.x(0)
qc.x(0)
qc.x(0)
qc.x(0)
qc.measure_all()

tic = time.time()
result = sim.run(qc).result()
toc = time.time()

t = toc - tic
print(t)

## 5X gate - parallel 
####################################

qc = QuantumCircuit(5)
qc.x(0)
qc.x(1)
qc.x(2)
qc.x(3)
qc.x(4)
qc.measure_all()

tic = time.time()
result = sim.run(qc).result()
toc = time.time()

t = toc - tic
print(t)

## 7X gate - parallel 
####################################

qc = QuantumCircuit(7)
qc.x(0)
qc.x(1)
qc.x(2)
qc.x(3)
qc.x(4)
qc.x(5)
qc.x(6)
qc.measure_all()

tic = time.time()
result = sim.run(qc).result()
toc = time.time()

t = toc - tic
print(t)

## 10X gate - parallel 
####################################

qc = QuantumCircuit(10)
qc.x(0)
qc.x(1)
qc.x(2)
qc.x(3)
qc.x(4)
qc.x(5)
qc.x(6)
qc.x(7)
qc.x(8)
qc.x(9)
qc.measure_all()

tic = time.time()
result = sim.run(qc).result()
toc = time.time()

t = toc - tic
print(t)