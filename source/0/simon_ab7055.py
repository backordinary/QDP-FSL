# https://github.com/parasol4791/quantumComp/blob/c1802ff9257ca99ee8536788290156adcc3a9221/algos/simon.py


import numpy as np
# importing Qiskit
from qiskit import IBMQ, Aer
from qiskit.providers.ibmq import least_busy
from qiskit import QuantumCircuit, transpile, assemble

# import basic plot tools
from qiskit.visualization import plot_histogram
from utils.backends import get_backend_aer, get_job_aer, get_job_ibmq


def simon_oracle(b):
   """returns a Simon oracle for bitstring b"""
   b = b[::-1]  # reverse b for easy iteration
   n = len(b)
   qc = QuantumCircuit(n * 2)
   # Do copy; |x>|0> -> |x>|x>
   for q in range(n):
      qc.cx(q, q + n)
   if '1' not in b:
      return qc  # 1:1 mapping, so just exit
   i = b.find('1')  # index of first non-zero bit in b
   # Do |x> -> |s.x> on condition that q_i is 1
   for q in range(n):
      if b[q] == '1':
         qc.cx(i, (q) + n)
   return qc

def step2_solveForB(n, counts):
   """This is the second (classical) step in Simon's problem.
      After getting m measurements z for counts, we need to solve a system of m x n equations
         zb = 0, i.e. bitwise dot product (mod 2) of our measurements with a secred string b all equal to zero"""

   #NOTE: this is not resolved!!!! Need to figure out how to frame a system of equations (a1 x1 + a2 x2 + ... + an xn) mod 2 = 0

   # Solve for hidden string b
   k = list(counts.keys())
   # Remove a trivial result with all zeros to avoid a singular matrix
   zeroRes = '0' * n
   k = [el for el in k if el != zeroRes]
   m = len(k)
   # Get tensors a, b for a system of linear equations ax = b
   a = np.ndarray((m, n))
   for i, count in enumerate(k):
      for j, elem in enumerate(count):
         a[i][j] = int(elem)
   # Make it square for a solver
   if m > n:
      pad = np.zeros((m, m - n))
      a = np.append(a, pad, axis=1)
   elif m < n:
      pad = np.zeros((n - m, n))
      a = np.append(a, pad, axis=0)

   b = np.zeros(max(n, m))

   bSolved = np.linalg.solve(a, b)
   print(bSolved)

if __name__ == "__main__":
   b = '0'
   n = len(b)

   sim = get_backend_aer()  # to enable statevector

   simon_circuit = QuantumCircuit(n * 2, n)

   # Apply Hadamard gates before querying the oracle
   simon_circuit.h(range(n))

   # Apply barrier for visual separation
   simon_circuit.barrier()

   simon_circuit += simon_oracle(b)

   # Apply barrier for visual separation
   simon_circuit.barrier()

   # Apply Hadamard gates to the input register
   simon_circuit.h(range(n))

   #simon_circuit.save_statevector()

   # Measure qubits
   simon_circuit.measure(range(n), range(n))
   simon_circuit.draw()

   # Simulator
   job = get_job_aer(simon_circuit)
   res = job.result()
   counts = res.get_counts()
   print(f"Simulation counts: {counts}")

   # b = 00
   # Simulation counts: {'10': 243, '00': 269, '11': 257, '01': 255}
   # b = 11
   # Simulation counts: {'00': 508, '11': 516}
   # b = 110
   # Out[1]: {'001': 253, '110': 254, '000': 218, '111': 299}


   # On real device
   job = get_job_ibmq(simon_circuit)
   res = job.result()
   print(f"Quantum counts: {res.get_counts()}")

   # b = 11 (can't do more than 5 qubits)
   # Quantum counts: {'00': 335, '01': 171, '10': 202, '11': 316}

   # b = 1110
   # Simulation counts: {'1010': 134, '0110': 132, '0001': 120, '1101': 116, '0111': 136, '0000': 122, '1011': 126, '1100': 138}

