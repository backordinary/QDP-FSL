# https://github.com/Nicholas-py/Quantummmmmmmmmmmmm/blob/d6b1f32f5f79adedda48eaea1c5613f54b35d3dc/dj.py
from setup import *
from qiskit import QuantumCircuit, QuantumRegister,ClassicalRegister
import random
def oracle(qc,qs):
  if random.choice(['balanced','never','EATMOREPIE']) == 'never':
    print('never')
    if random.randint(0,1) == 0:
      qc.x(-1)
  else:
    print('balanced')
    for i in range(qs):
      if random.randint(0,1) == 0:
        #qc.x(i)
        pass
    qc.cx(0,qs)

def dj(qs):
  x = QuantumRegister(qs)
  o = QuantumRegister(1)
  op = ClassicalRegister(qs)
  qc = QuantumCircuit(x,o,op)
  qc.x(qs)
  h_it_all(qc)
  oracle(qc,qs)
  h_it_all(qc)
  circuitmeasure(qc,op,x)
  simulate(qc)

