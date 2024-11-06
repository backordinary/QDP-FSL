# https://github.com/msqc-goethe/quantum-boolean-functions/blob/4f11b168245074e8f4a4be8d9e1fe3080a27e866/Qiskit/qiskit_example.py
from qiskit import *
from qiskit.compiler import transpile
from qiskit.circuit.classicalfunction import BooleanExpression

op = BooleanExpression('a & b', name='AND')
circ = QuantumCircuit(3)
circ.append(op,range(3))
transpiled_circuit = transpile(circ, basis_gates=['rx', 'rz','ry' ,'p','cx'])


print(transpiled_circuit.count_ops())
transpiled_circuit.draw('mpl', filename='and_example')