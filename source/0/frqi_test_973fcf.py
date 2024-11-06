# https://github.com/Toil12/QuamtumThesis/blob/3443729cd39957ad89f5dddf6ccbdfdeca0a6c13/SystemTest/FRQI_test.py
# Importing standard Qiskit libraries and configuring account
import qiskit as qk
from qiskit import QuantumCircuit, Aer, IBMQ
from qiskit import transpile, assemble
from math import pi
from qiskit.visualization import plot_histogram
theta1 = pi/2 # all pixels black
theta2 = pi/2# 01
theta3 = pi/2# 11
theta4 = pi/2
qc = QuantumCircuit(3)


qc.h(0)
qc.h(1)

qc.barrier()
#Pixel 1

qc.cry(theta1,0,2)
qc.cx(0,1)
qc.cry(-theta1,1,2)
qc.cx(0,1)
qc.cry(theta1,1,2)

qc.barrier()
#Pixel 2

qc.x(1)
qc.cry(theta2,0,2)
qc.cx(0,1)
qc.cry(-theta2,1,2)
qc.cx(0,1)
qc.cry(theta2,1,2)

qc.barrier()

qc.x(1)
qc.x(0)
qc.cry(theta3,0,2)
qc.cx(0,1)
qc.cry(-theta3,1,2)
qc.cx(0,1)
qc.cry(theta3,1,2)


qc.barrier()

qc.x(1)

qc.cry(theta4,0,2)
qc.cx(0,1)
qc.cry(-theta4,1,2)
qc.cx(0,1)
qc.cry(theta4,1,2)

qc.measure_all()

print(qc)

aer_sim = Aer.get_backend('aer_simulator')
t_qc = transpile(qc, aer_sim)
qobj = assemble(t_qc, shots=4096)
result = aer_sim.run(qobj).result()
counts = result.get_counts(qc)
print(counts)
plot_histogram(counts)