# https://github.com/MSwenne/BEP/blob/f2848e3121e976540fb10171fdfbc6670dd28459/Code/site-packages/qiskit/extensions/standard/mcx.py
import numpy as np
import math

from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit import ClassicalRegister
from qiskit import IBMQ, execute, Aer
import qiskit as qk
IBMQ.load_accounts()
IBMQ.backends()

n=4 

def main():
    ## circuit generation
    q = QuantumRegister( n )
    c = ClassicalRegister( n )
    qc = QuantumCircuit(q, c)

    qc.swap(q[0],q[3])
    qc.swap(q[1],q[2])

    mcx_r(qc, q, 0, 2, 3, [0,1,2])

    qc.swap(q[1],q[2])
    qc.swap(q[0],q[3])

    backend = qk.BasicAer.get_backend('unitary_simulator')
    job = execute(qc, backend=backend, shots=1024, max_credits=3)
    print(np.around(job.result().get_unitary().real,2))


    qc.measure(q, c)

    backend = qk.BasicAer.get_backend('qasm_simulator')
    # backend = qk.BasicAer.get_backend('statevector_simulator')
    job = execute(qc, backend=backend)
    print(job.result().get_counts())
    print(qc)

def mcx(qc, q, a, b, c, index):
	mcx_r(qc, q, a, b, c, index)

def mcx_r(qc, q, a, b, c, index):
	if a == b:
		return
	if a in index and b in index:
		if a+1 == b:
			qc.ccx(q[a],q[b],q[c])
		else:
			qc.barrier(q[a],q[b],q[c])
			qc.h(q[c])
			mcx_r(qc,q,a,b-1,c, index)
			qc.tdg(q[c])
			qc.cx(q[b], q[c])
			qc.t(q[c])
			mcx_r(qc,q,a,b-1,c, index)
			qc.tdg(q[c])
			qc.cx(q[b], q[c])
			qc.t(q[c])
			qc.h(q[c])
			qc.barrier(q[a],q[b],q[c])
			incrementer(qc,q,a,b, index)
			qc.barrier()
			j = 0
			for i in range(b, a, -1):
				if i in index:
					qc.u1(-math.pi/(pow(2, b-i-j+2)), q[i])
				else:
					j = j+1
			qc.barrier()
			decrementer(qc,q,a,b, index)
			qc.barrier()
			j = 0
			for i in range(b, a, -1):
				if i in index:
					qc.u1(math.pi/(pow(2, b-i-j+2)), q[i])
				else:
					j = j+1
			qc.u1(math.pi/(pow(2, b-a-j+1)), q[a])
			qc.barrier()
	elif a not in index:
		mcx_r(qc,q,a+1,b,c, index)
	elif b not in index:
		mcx_r(qc,q,a,b-1,c, index)


def incrementer(qc, q, a, b, index):
	if len(index) > 2:
		for i in range(len(index)-1,1,-1):
			if a <= index[i] and index[i] <=  b:
				mcx(qc, q, int(index[0]), int(index[i-1]), int(index[i]), index)
	if len(index) > 1:
		qc.cx(q[int(index[0])],q[int(index[1])])
	qc.x(q[int(index[0])])

def decrementer(qc, q, a, b, index):
	qc.x(q[int(index[0])])
	if len(index) > 1:
		qc.cx(q[int(index[0])],q[int(index[1])])
	if len(index) > 2:
		for i in range(2,len(index)):
			if a <= index[i] and index[i] <=  b:
				mcx(qc, q, int(index[0]), int(index[i-1]), int(index[i]), index)
main()