# https://github.com/pc2/rdmft_on_qc/blob/00a461db1fe12e771e144f854ab65316d3b015d1/utils/unitary_matrix.py
import numpy as np
from qiskit import *
from qiskit import Aer
from qiskit.circuit import Gate
from math import pi
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Unroller

#Changing the simulator 
#backend = Aer.get_backend('unitary_simulator')
backend = Aer.get_backend('statevector_simulator')

#The circuit without measurement

theta = 0.3 # theta can be anything (pi chosen arbitrarily)
circ = QuantumCircuit(4)

circ.x(0)
circ.x(1)
circ.x(2)
circ.draw()
print(circ)
print(circ.decompose())

#job execution and getting the result as an object
job = execute(circ, backend)
result = job.result()

#pass_ = Unroller(['u1', 'u2', 'u3', 'cx'])
#pm = PassManager(pass_)
#new_circ = pm.run(circ)
#print(new_circ)

#get the unitary matrix from the result object
#u1=result.get_unitary(circ, decimals=3)
v1=result.get_statevector(circ, decimals=3)

circ = QuantumCircuit(4)

circ.x(0)
circ.x(1)
circ.x(3)
circ.draw()
print(circ)

#job execution and getting the result as an object
job = execute(circ, backend)
result = job.result()

#get the unitary matrix from the result object
#u2=result.get_unitary(circ, decimals=3)
v2=result.get_statevector(circ, decimals=3)
#print(u1)
#print(u2)
#print(u1-u2)


circ = QuantumCircuit(4)

circ.x(0)
circ.x(1)
circ.x(3)
circ.h(2)
circ.cnot(2,3)
circ.draw()
print(circ)

#job execution and getting the result as an object
job = execute(circ, backend)
result = job.result()

#get the unitary matrix from the result object
#u2=result.get_unitary(circ, decimals=3)
v3=result.get_statevector(circ, decimals=3)

for i in range(len(v1)):
    print(i,v1[i],v2[i],v3[i])

pass_ = Unroller(['u1', 'u2', 'u3', 'cx'])
pm = PassManager(pass_)
new_circ = pm.run(circ)
print(new_circ)
