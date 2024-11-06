# https://github.com/1chooo/data-structure-and-algorithms/blob/24ba2e4bf47cd2f77bb4e0ea1adf39d6906427a0/Algorithms/quantum/CH01/prog_03.py
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit

qrx = QuantumRegister(3, 'x')
qry = QuantumRegister(2, 'y')
qrz = QuantumRegister(1, 'z')
cr = ClassicalRegister(4, 'c')
qc = QuantumCircuit(qrx,qry,qrz,cr) 
qc.measure([qrx[1], qrx[2]], [cr[0], cr[1]]) 
qc.measure([4, 5], [2, 3])
qc.draw()