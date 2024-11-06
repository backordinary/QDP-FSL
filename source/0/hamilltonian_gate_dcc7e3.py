# https://github.com/rohitganti/quanutm_computing-/blob/235dd44ad64de21194faac184b584d19559b66c3/hamilltonian_gate.py
import numpy as np
from qiskit import *

from qiskit.circuit import Gate
my_gate= Gate(name='my_gate',num_qubits=2, params=[])
qr= QuantumRegister(3,'q')
circ= QuantumCircuit(qr)
circ.append(my_gate,[qr[0],qr[1]])
circ.append(my_gate, [qr[1], qr[2]])
circ.draw()


sub_q = QuantumRegister(2)
sub_circ = QuantumCircuit(sub_q, name='sub_circ')
sub_circ.h(sub_q[0])
sub_circ.crz(1, sub_q[0], sub_q[1])
sub_circ.barrier()
sub_circ.id(sub_q[1])
sub_circ.u3(1, 2, -2, sub_q[0])
sub_inst = sub_circ.to_instruction()



qr = QuantumRegister(3, 'q')
circ = QuantumCircuit(qr)
circ.h(qr[0])
circ.cx(qr[0], qr[1])
circ.cx(qr[1], qr[2])
circ.append(sub_inst, [qr[1], qr[2]])

circ.draw()

decomposed_circ = circ.decompose()
decomposed_circ.draw()
#this is for drawign my circuit out without changing the orignial one rather jsut spillting them as many

from qiskit.circuit import Parameter

theta = Parameter('θ')
n = 5
qc = QuantumCircuit(5, 1)
qc.h(0)
for i in range(n-1):
    qc.cx(i, i+1)
qc.barrier()
qc.rz(theta, range(5))
qc.barrier()
for i in reversed(range(n-1)):
    qc.cx(i, i+1)
qc.h(0)
qc.measure(0, 0)
qc.draw('mpl')


