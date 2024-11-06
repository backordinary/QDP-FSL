# https://github.com/one-marker/qiskitIBM/blob/90a3ad246aae4c296b8aad26ac0c99ca83b660f6/main.py
import numpy as np
import math
import Teleport
import Grv

from qiskit import(
  QuantumCircuit,
  QuantumRegister,
  ClassicalRegister,
  Aer,
  execute)
from qiskit.visualization import plot_histogram


print("TELEPORT")
print(Teleport.build().draw())

print("GROVER")
print(Grv.build().draw())