# https://github.com/nemkin/tdk-2022-codes/blob/939d3ef5c8e9c2b5952326d770d35a61005e206e/mem_calc.py
import sys
from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit, execute, Aer
from qiskit.circuit.library import SwapGate
from qiskit.circuit.library.arithmetic import WeightedAdder

def human(bytes):
  iters = 0
  names = ["B", "KB", "MB", "GB", "TB", "PB"]

  i = 0;
  while (i < len(names) - 1) and ((bytes >> 10) > 0):
    bytes = bytes >> 10
    i += 1

  color = ""
  if i > 3:
    color = "\color{red}"

  return f"{color} {bytes} {names[i]}"

def header():
  print("n & ", end="")
  print("Qubitek & ", end="")
  print("Regiszer & ", end="")
  print("Operátor", end="")
  print(" \\\\")
  print("\hline")

def calc(n, last=False):
  qubits = 6*(n-1)
  bytes =  16

  reg_count = 1 << qubits
  reg_size = reg_count * bytes

  op_count = reg_count * reg_count
  op_size = op_count * bytes

  print(n, end=" & ")
  print(qubits, end=" & ")
  print(human(reg_size), end=" & ")
  print(human(op_size), end="")
  if not last:
    print(" \\\\")
  else:
    print()

header()
lastnum = 7
for i in range(2, lastnum+1):
  calc(i, (i == lastnum))

