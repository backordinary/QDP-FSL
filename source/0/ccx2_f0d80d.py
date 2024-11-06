# https://github.com/Namr/YAQCS/blob/65a286f05fc917a39462d23ab1ad368f84466e7c/BQSKit%20Extensions/examples/ccx2.py
#https://journals.aps.org/pra/pdf/10.1103/PhysRevA.79.012312
#pg 4 VI.C bottom left
import numpy as np
from qiskit import *
from qiskit import Aer
from qiskit.circuit.library import GMS

backend = Aer.get_backend('unitary_simulator')

XXX1 = GMS(num_qubits=3, theta=[[0, np.pi, np.pi],
                                [0, 0, np.pi],
                                [0, 0, 0]])
XXX2 = GMS(num_qubits=3, theta=[[0, np.pi/2, np.pi/2],
                                [0, 0, np.pi/2],
                                [0, 0, 0]])
XXXM2 = GMS(num_qubits=3, theta=[[0, -np.pi/2, -np.pi/2],
                                 [0, 0, -np.pi/2],
                                 [0, 0, 0]])
XXX4 = GMS(num_qubits=3, theta=[[0, np.pi/4, np.pi/4],
                                [0, 0, np.pi/4],
                                [0, 0, 0]])
XXXM4 = GMS(num_qubits=3, theta=[[0, -np.pi/4, -np.pi/4],
                                [0, 0, -np.pi/4],
                                [0, 0, 0]])

qr = QuantumRegister(3, 'q')
qc = QuantumCircuit(qr)
qc.ccx(2,1,0) #LSB is target bit

#CNOT is its own inverse, see if MS-based CNOT works
qc.ry(np.pi/2,2)
qc.ry(np.pi/2,1)
qc.ry(np.pi/2,0)
qc.p(np.pi/4,0) #rz
qc.append(XXX2, qr)

qc.rx(-np.pi/2,2)
qc.rx(-np.pi/2,1)
qc.rx(-np.pi/2,0)
qc.p(-np.pi/2,0) #rz
qc.rx(-np.pi/4,2)
qc.rx(-np.pi/4,1)
qc.rx(-np.pi/4,0)
qc.append(XXX4, qr)

qc.p(np.pi/2,0) #rz
qc.append(XXX2, qr)

qc.rx(np.pi/2,2)
qc.rx(np.pi/2,1)
qc.rx(np.pi/2,0)
qc.ry(-np.pi/2,2)
qc.ry(-np.pi/2,1)
qc.ry(-np.pi/2,0)

#no, we don't get ID matrix back
#but it's close, likely the global phase again
#qc.p(np.pi/4,0) #rz
#qc.x(0)
#qc.p(np.pi/4,0) #rz
#qc.x(0)

#qc.p(-3*np.pi/4,0) #rz
#qc.ry(-np.pi/2,1)

job = execute(qc, backend)
result = job.result()
print(result.get_unitary(qc, decimals=3))
