# https://github.com/cbarange/workshop_cybersecu_quantum/blob/4593adf8d6f8a93b2b2b9ecec53ce770c4ec5142/shor_qiskit.py
#!/bin/python3

import math
import numpy as np
from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit.algorithms import Shor
import sys
import time




def shor_qiskit(number):
  N = int(number)
  backend = Aer.get_backend('aer_simulator')
  quantum_instance = QuantumInstance(backend, shots=1024)
  shor = Shor(quantum_instance=quantum_instance)
  result = shor.factor(N)
  return result

  
print("number;factors;compute_time;algo;prime")


primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
for number in range(3,16):
  if number%2==0:
    continue
  start_time = time.time()
  res = shor_qiskit(number)
  compute_time = "%s" % (time.time() - start_time)

  factors = map(str,res.factors)
  print(f"{number};({'' if len(res.factors)==0 else ','.join(factors)});{compute_time};qiskit;{'prime' if number in primes else 'non-prime'}")