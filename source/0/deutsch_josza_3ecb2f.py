# https://github.com/StevenSchuerstedt/QuantumComputing/blob/0b32d1c642450aec87b2fa0204ba285136875e6b/code/deutsch_josza.py
###Deutsch's Josza's Algorithm
######################
from qiskit.aqua.algorithms import VQE, NumPyEigensolver
from qiskit import QuantumCircuit, Aer
import qiskit.quantum_info as qi
from qiskit.quantum_info import Operator
import numpy as np

n = 4

def createOracle(size):

    #constant oracle
    #f(x) = 0
    #return np.identity(2**(size+1))

    #balanced function
    #only check for one bit, to decide for f = 1/0
    # start with identity matrix
    size = 2 **(size+1)
    a = np.identity(size)

    # swap correct rows
    # note: qiskit little endian convention
    for n in range(0, int(size / 2), 2):
        i, j = 1 + n, 1 + n + int(size / 2)
        a.T[[i, j]] = a.T[[j, i]]
    return a

##create arbitrary balanced function
##specifiy list of points where f should evaluate to 1
##eg. f(1), f(3), ...
##start with identity matrix (when f = 0, nothing changes)
##switch columns of matrix

sim = Aer.get_backend('aer_simulator')

qc = QuantumCircuit(n+1, n)

#put last qubit in state |1>
qc.x(n)
qc.barrier()

# apply h-gates, |+>
for q in range(n):
    qc.h(q)

# apply h-gate to last qubit |->
# note: eigenstate of U_f (phase kickback)
qc.h(n)

qc.barrier()

# apply oracle
U_f = createOracle(n)

l = []
for i in range(n+1):
    l.append(i)

qc.unitary(U_f, l, label='U_f')

qc.barrier()


# apply h-gate again to collapse
for q in range(n):
    qc.h(q)

qc.barrier()

#qc = Operator(qc.reverse_bits())
for i in range(n):
    qc.measure(i, i)

print(qc)

result = sim.run(qc).result()
counts = result.get_counts()
print(counts)


## all 0s => function is constant
## one or more 1 => function is balanced