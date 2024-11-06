# https://github.com/gp-maxwell/Shors-Algorithm/blob/d54658b196549e200b6af8e5f5685ae2291090a7/fp51.py
import qiskit as q
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, BasicAer, IBMQ, execute
from numpy import pi
from qiskit.visualization import plot_histogram
import math

N = 51

qb = 8
qr = q.QuantumRegister(qb)
cr = q.ClassicalRegister(qb/2)

circuit = q.QuantumCircuit(qr, cr)
circuit.h(0)
circuit.h(1)
circuit.h(2)
circuit.h(3)

circuit.barrier()

# Modular Exponentiation

circuit.cx(3,7)
circuit.cx(2,6)
circuit.barrier()

# Inverse QFT
circuit.h(0)

circuit.crz(-np.pi/2,[0],[1])
circuit.h(1)

circuit.crz(-np.pi/2,[1],[2])
circuit.crz(-np.pi/4,[0],[2])
circuit.h(2)

circuit.crz(-np.pi/2,[2],[3])
circuit.crz(-np.pi/4,[1],[3])
circuit.crz(-np.pi/8,[0],[3])
circuit.h(3)

circuit.barrier()
circuit.measure([0], [0])
circuit.measure([1], [1])
circuit.measure([2], [2])
circuit.measure([3], [3])

# Execute Simulation
backend = BasicAer.get_backend('qasm_simulator')
shots = 15000
results = execute(circuit, backend=backend, shots=shots).result()
counts = results.get_counts()
print(counts)

print(circuit)

a = 4
A = int(max(counts, key=counts.get),2)
if A == 0:
    print('Division by 0, stopping script')
    quit()
# print(A)
r = int((2**(qb/2))/A)
P = int(a**(r/2)+1)
Q = int(a**(r/2)-1)

f1 = math.gcd(P,N)
f2 = math.gcd(Q,N)

print(f'Prime factors of {N} are: {f1} and {f2}')
