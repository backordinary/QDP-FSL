# https://github.com/Eshan-Yadav/quantum-computing-for-string-matching/blob/d32d5db3ed41d2c6520f09211d00259fbc01a34c/0000oracle.py
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute,IBMQ
import math
from qiskit.tools.monitor import job_monitor

# IBMQ.enable_account('Enter API token')
# provider = IBMQ.get_provider(hub='ibm-q')
        
pi = math.pi
q = QuantumRegister(4,'q')
c = ClassicalRegister(4,'c')
qc = QuantumCircuit(q,c)

print('\nInitialising Circuit...\n')

### Initialisation ###

qc.h(q[0])
qc.h(q[1])
qc.h(q[2])
qc.h(q[3])

print('\nPreparing Oracle circuit.... for 0000\n')

### 0000 Oracle ###

### 0001 Oracle ###

qc.x(q[1])
qc.x(q[2])
qc.x(q[3])

qc.cp(pi/4, q[0], q[3])
qc.cx(q[0], q[1])
qc.cp(-pi/4, q[1], q[3])
qc.cx(q[0], q[1])
qc.cp(pi/4, q[1], q[3])
qc.cx(q[1], q[2])
qc.cp(-pi/4, q[2], q[3])
qc.cx(q[0], q[2])
qc.cp(pi/4, q[2], q[3])
qc.cx(q[1], q[2])
qc.cp(-pi/4, q[2], q[3])
qc.cx(q[0], q[2])
qc.cp(pi/4, q[2], q[3])

qc.x(q[1])
qc.x(q[2])
qc.x(q[3])

print(qc.draw())


