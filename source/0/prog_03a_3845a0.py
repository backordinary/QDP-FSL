# https://github.com/1chooo/data-structure-and-algorithms/blob/24ba2e4bf47cd2f77bb4e0ea1adf39d6906427a0/Algorithms/quantum/CH05/prog_03a.py
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit

qrx = QuantumRegister(3, 'x')
qry = QuantumRegister(1, 'y')
cr = ClassicalRegister(3, 'c')
qc = QuantumCircuit(qrx, qry, cr)

qc.h(qrx)
qc.x(qry)
qc.h(qry)
qc.barrier()
qc.x(qry)
qc.barrier()
qc.h(qrx)
qc.h(qry)
qc.measure(qrx, cr)

qc.draw("mpl")