# https://github.com/1chooo/Quantum-Algorithm/blob/c61170ed839c4f1f92880b28cc4c69fc877ca5ae/CH05/prog_05a.py
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit

qrx = QuantumRegister(3, 'x')
qry = QuantumRegister(1, 'y')
cr = ClassicalRegister(3, 'c')
qc = QuantumCircuit(qrx, qry, cr)
qc.h(qrx)
qc.x(qry)       
# qc.h(qry) 機率會不同
qc.h(qry)
qc.barrier()
qc.cx(qrx[0], qry)
qc.barrier()
qc.h(qrx)
qc.h(qry)
qc.measure(qrx, cr)

qc.draw("mpl")