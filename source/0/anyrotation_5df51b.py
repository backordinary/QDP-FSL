# https://github.com/vicvaleeva/qiskit-learn/blob/e8d179369e6c3f4ba815604f8183cec5044a8c7a/BuildGates/anyRotation.py
from qiskit import QuantumCircuit
from qiskit.circuit import Gate

A = Gate('A', 1, [])
B = Gate('B', 1, [])
C = Gate('C', 1, [])
alpha = 1

qc = QuantumCircuit(2)
qc.append(C, [1])
qc.cz(0, 1)
qc.append(B, [1])
qc.cz(0, 1)
qc.append(A, [1])
qc.u1(alpha, 0)

print(qc)