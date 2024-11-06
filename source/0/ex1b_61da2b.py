# https://github.com/Ecaterina-Hrib/Quantum-Computing/blob/9daff5b15c025ac2e7de57c2ba7b636188a706b1/tema%202/ex1b.py
#functia f(x)=(a^x)%15
from qiskit import *
from math import pi
from qiskit.circuit.library import QFT
import numpy as np

a=7

circuit=QuantumCircuit(12,12)
circuit.append(QFT(8,True),circuit.qubits[:8])

def modulo15(a, power):
    U = QuantumCircuit(4)        
    for iteration in range(power):
        if a in [2,13]:
            U.swap(0,1)
            U.swap(1,2)
            U.swap(2,3)
        if a in [7,8]:
            U.swap(2,3)
            U.swap(1,2)
            U.swap(0,1)
        if a == 11:
            U.swap(1,3)
            U.swap(0,2)
        if a in [7,11,13]:
            for q in range(4):
                U.x(q)
    U = U.to_gate()
    U.name = "%i^%i mod 15" % (a, power)
    c_U = U.control()
    return c_U
circuit.append(modulo15(a, 2**0), 
         [0] + [x+8 for x in range(4)])
circuit.append(modulo15(a, 2**1), 
         [1] + [x+8 for x in range(4)])
circuit.append(modulo15(a, 2**2), 
         [2] + [x+8 for x in range(4)])
circuit.append(modulo15(a, 2**3), 
         [3] + [x+8 for x in range(4)])
circuit.append(modulo15(a, 2**4), 
         [4] + [x+8 for x in range(4)])
circuit.append(modulo15(a, 2**5), 
         [5] + [x+8 for x in range(4)])
circuit.append(modulo15(a, 2**6), 
         [6] + [x+8 for x in range(4)])
circuit.append(modulo15(a, 2**7), 
         [7] + [x+8 for x in range(4)])

circuit.append(QFT(8,True),circuit.qubits[:8])
circuit.measure(0,0)
circuit.measure(1,1)
circuit.measure(2,2)
circuit.measure(3,3)
circuit.measure(4,4)
circuit.measure(5,5)
circuit.measure(6,6)
circuit.measure(7,7)

print(circuit.draw())

#pentru a vedea rezultatul
simulator=Aer.get_backend('qasm_simulator')
job=execute(circuit,simulator)
result=job.result()
counts=result.get_counts()
print(counts)