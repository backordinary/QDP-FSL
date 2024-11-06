# https://github.com/Ecaterina-Hrib/Quantum-Computing/blob/9daff5b15c025ac2e7de57c2ba7b636188a706b1/lab12/ex1.py
from qiskit import *
from qiskit.circuit.library import *
from qiskit.visualization import plot_histogram
#grover 101 110

n=3
nr=n
cr=n
c=QuantumCircuit(nr,cr)
ccz = MCMT(ZGate(), 2, 1)
ccz.name="CCZ"

c.h(0)
c.h(1)
c.h(2)
c.cz(1,0)
c.cz(2,0)
c.h(0)
c.h(1)
c.h(2)
c.x(0)
c.x(1)
c.x(2)
c.append(ccz, [2, 1, 0])
c.x(0)
c.x(1)
c.x(2)
c.h(0)
c.h(1)
c.h(2)
for i in range(n):
	c.measure(i,i)

print(c.draw())
simulator=Aer.get_backend('qasm_simulator')
job=execute(c,simulator)
result=job.result()
counts=result.get_counts()
print(counts)
#groover restul

c2=QuantumCircuit(nr,cr)
c2.h(0)
c2.h(1)
c2.h(2)
c2.cz(1,0)
c2.cz(2,0)
c2.append(ccz, [2, 1, 0])
c2.h(0)
c2.h(1)
c2.h(2)
c2.x(0)
c2.x(1)
c2.x(2)
c2.append(ccz, [2, 1, 0])
c2.x(0)
c2.x(1)
c2.x(2)
c2.h(0)
c2.h(1)
c2.h(2)
for i in range(n):
	c2.measure(i,i)
print(c2.draw())
simulator2=Aer.get_backend('qasm_simulator')
job2=execute(c2,simulator2)
result2=job2.result()
counts2=result2.get_counts()
print(counts2)



