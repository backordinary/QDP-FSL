# https://github.com/erenaykrcn/HAE/blob/69bae03f79cf6cad7c973fd23c8d3f675523781a/modules/qnn/qcircuits/pqc3.py
from qiskit import QuantumCircuit


def pqc3(x, theta):
	qc = QuantumCircuit(4)
	qc.rx(x[0], 0)
	qc.rx(x[1], 1)
	qc.rx(x[2], 2)
	qc.rx(x[3], 3)

	qc.ry(theta[0], 0)
	qc.ry(theta[1], 1)
	qc.ry(theta[2], 2)
	qc.ry(theta[3], 3)

	qc.rx(theta[4], 0)
	qc.rx(theta[5], 1)
	qc.rx(theta[6], 2)
	qc.rx(theta[7], 3)

	qc.cx(3, 0)
	qc.cx(0, 1)
	qc.cx(1, 2)
	qc.cx(2, 3)
	return qc
	
