# https://github.com/nkpro2000/IVyearProject/blob/678af0a893b9d6d11af0d05f0babab331f517ee5/qiskit_.py
from functools import partial

import qiskit
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import BasicAer, IBMQ
from qiskit.tools.monitor import job_monitor

SHOTS = 52

sbackend = BasicAer.get_backend('qasm_simulator')
qaccount = qbackend1 = qbackend2 = None

def execute(circute, backend=sbackend, shots=None, memory=True):
    if shots is None:
        shots = SHOTS
    return qiskit.execute(circute, backend, shots=shots, memory=memory)

def result(job, moc=2):
    job_monitor(job)
    r = job.result()

    if moc == 0:
        return r
    elif moc == 1:
        return r.get_counts()
    elif moc == 2:
        return r.get_memory()
    elif moc == 3:
        return r.get_memory(), r.get_counts()
    else:
        return r, r.get_memory(), r.get_counts()


'''
>>> for i in range(32):
...  print(' ' if i%3 else '>', str(i).zfill(2), i%3, bin(i)[2:].zfill(5), end=' ')
...  b=bin(i)[2:].zfill(5)
...  a,aa = (sum(map(int,b[::2])), sum(map(int,b[1::2])))
...  print(a, aa, abs(a-aa), '' if i%3 else '<')
... 
> 00 0 00000 0 0 0 <
  01 1 00001 1 0 1 
  02 2 00010 0 1 1 
> 03 0 00011 1 1 0 <
  04 1 00100 1 0 1 
  05 2 00101 2 0 2 
> 06 0 00110 1 1 0 <
  07 1 00111 2 1 1 
  08 2 01000 0 1 1 
> 09 0 01001 1 1 0 <
  10 1 01010 0 2 2 
  11 2 01011 1 2 1 
> 12 0 01100 1 1 0 <
  13 1 01101 2 1 1 
  14 2 01110 1 2 1 
> 15 0 01111 2 2 0 <
  16 1 10000 1 0 1 
  17 2 10001 2 0 2 
> 18 0 10010 1 1 0 <
  19 1 10011 2 1 1 
  20 2 10100 2 0 2 
> 21 0 10101 3 0 3 <
  22 1 10110 2 1 1 
  23 2 10111 3 1 2 
> 24 0 11000 1 1 0 <
  25 1 11001 2 1 1 
  26 2 11010 1 2 1 
> 27 0 11011 2 2 0 <
  28 1 11100 2 1 1 
  29 2 11101 3 1 2 
> 30 0 11110 2 2 0 <
  31 1 11111 3 2 1 
>>>              '''
