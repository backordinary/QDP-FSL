# https://github.com/sdabi/quantum_mini_project/blob/5dd01ced7638ad89fd319644401e68183d19d81f/mini_project_code.py
#!/usr/bin/env python
# coding: utf-8

from qiskit import *
from qiskit.tools.visualization import plot_histogram
from qiskit.tools.monitor import job_monitor

# simulator instance
sim = Aer.get_backend('qasm_simulator')

# circle creation:
# applying Hadamrd on every input qubit, and creating |-> state as input.
# creating function logic - "and between all qubits", and useing phase kickback
# applying Hadamrd on every ouput qubit
# measure qubit number 0
circs = []
circ = QuantumCircuit((2*2),2)
circ.h(0)
for i in range(2-1):
    circ.h(1+(i*2))
circ.x((2*2)-1)
circ.h((2*2)-1)
circ.barrier()
for i in range(0,(2*2)-2,2):
    circ.ccx(i,i+1,i+2)
circ.cx((2*2)-2,(2*2)-1)
for i in range((2*2)-2,0,-2):
    circ.ccx(i-2,i-1,i)
circ.barrier()
circ.h(0)
for i in range(2-1):
    circ.h(1+(i*2))
circ.measure(0,0)
circs.append(circ)
circ = QuantumCircuit((3*2),3)
circ.h(0)
for i in range(3-1):
    circ.h(1+(i*2))
circ.x((3*2)-1)
circ.h((3*2)-1)
circ.barrier()
for i in range(0,(3*2)-2,2):
    circ.ccx(i,i+1,i+2)
circ.cx((3*2)-2,(3*2)-1)
for i in range((3*2)-2,0,-2):
    circ.ccx(i-2,i-1,i)
circ.barrier()
circ.h(0)
for i in range(3-1):
    circ.h(1+(i*2))
circ.measure(0,0)
circs.append(circ)
circ = QuantumCircuit((4*2),4)
circ.h(0)
for i in range(4-1):
    circ.h(1+(i*2))
circ.x((4*2)-1)
circ.h((4*2)-1)
circ.barrier()
for i in range(0,(4*2)-2,2):
    circ.ccx(i,i+1,i+2)
circ.cx((4*2)-2,(4*2)-1)
for i in range((4*2)-2,0,-2):
    circ.ccx(i-2,i-1,i)
circ.barrier()
circ.h(0)
for i in range(4-1):
    circ.h(1+(i*2))
circ.measure(0,0)
circs.append(circ)
circ = QuantumCircuit((5*2),5)
circ.h(0)
for i in range(5-1):
    circ.h(1+(i*2))
circ.x((5*2)-1)
circ.h((5*2)-1)
circ.barrier()
for i in range(0,(5*2)-2,2):
    circ.ccx(i,i+1,i+2)
circ.cx((5*2)-2,(5*2)-1)
for i in range((5*2)-2,0,-2):
    circ.ccx(i-2,i-1,i)
circ.barrier()
circ.h(0)
for i in range(5-1):
    circ.h(1+(i*2))
circ.measure(0,0)
circs.append(circ)
circ = QuantumCircuit((6*2),6)
circ.h(0)
for i in range(6-1):
    circ.h(1+(i*2))
circ.x((6*2)-1)
circ.h((6*2)-1)
circ.barrier()
for i in range(0,(6*2)-2,2):
    circ.ccx(i,i+1,i+2)
circ.cx((6*2)-2,(6*2)-1)
for i in range((6*2)-2,0,-2):
    circ.ccx(i-2,i-1,i)
circ.barrier()
circ.h(0)
for i in range(6-1):
    circ.h(1+(i*2))
circ.measure(0,0)
circs.append(circ)
circ = QuantumCircuit((7*2),7)
circ.h(0)
for i in range(7-1):
    circ.h(1+(i*2))
circ.x((7*2)-1)
circ.h((7*2)-1)
circ.barrier()
for i in range(0,(7*2)-2,2):
    circ.ccx(i,i+1,i+2)
circ.cx((7*2)-2,(7*2)-1)
for i in range((7*2)-2,0,-2):
    circ.ccx(i-2,i-1,i)
circ.barrier()
circ.h(0)
for i in range(7-1):
    circ.h(1+(i*2))
circ.measure(0,0)
circs.append(circ)
circ = QuantumCircuit((8*2),8)
circ.h(0)
for i in range(8-1):
    circ.h(1+(i*2))
circ.x((8*2)-1)
circ.h((8*2)-1)
circ.barrier()
for i in range(0,(8*2)-2,2):
    circ.ccx(i,i+1,i+2)
circ.cx((8*2)-2,(8*2)-1)
for i in range((8*2)-2,0,-2):
    circ.ccx(i-2,i-1,i)
circ.barrier()
circ.h(0)
for i in range(8-1):
    circ.h(1+(i*2))
circ.measure(0,0)
circs.append(circ)
circ = QuantumCircuit((9*2),9)
circ.h(0)
for i in range(9-1):
    circ.h(1+(i*2))
circ.x((9*2)-1)
circ.h((9*2)-1)
circ.barrier()
for i in range(0,(9*2)-2,2):
    circ.ccx(i,i+1,i+2)
circ.cx((9*2)-2,(9*2)-1)
for i in range((9*2)-2,0,-2):
    circ.ccx(i-2,i-1,i)
circ.barrier()
circ.h(0)
for i in range(9-1):
    circ.h(1+(i*2))
circ.measure(0,0)
circs.append(circ)

circs[1].draw()

# running each circle 1 Million times
results = []
for circ in circs:
    res = execute(circ,backend=sim, shots=1024*1024).result()
    results.append(res)

plot_histogram(results[1].get_counts())
plot_histogram(results[1].get_counts())
plot_histogram(results[2].get_counts())
plot_histogram(results[3].get_counts())
plot_histogram(results[4].get_counts())
plot_histogram(results[5].get_counts())
plot_histogram(results[6].get_counts())
plot_histogram(results[7].get_counts())
