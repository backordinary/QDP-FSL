# https://github.com/MoizAhmedd/quantumtesting/blob/e0643e26628ef1cdc6d0c7309264bddd38004905/firstscore.py
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer

q = QuantumRegister(2)
c = ClassicalRegister(2)

#Building Circuits
my_first_score = QuantumCircuit(q,c)

#Pauli Operations
my_first_score.x(q[0])
my_first_score.y(q[1])
my_first_score.barrier(q)

#Clifford Operations, can be referred to as superposition generating operations
my_first_score.h(q)
my_first_score.s(q[0])
my_first_score.s(q[1].inverse())
my_first_score.cx(q[0],q[1])
my_first_score.barrier(q)

#Non-Clifford Operations
my_first_score.t(q[0])
my_first_score.t(q[1]).inverse()
my_first_score.barrier(q)

#Measurement Operations
my_first_score.measure(q,c)

#Execute The Circuit
job = execute(my_first_score,backend= Aer.get_backend('qasm_simulator'),shots=1024)
result = job.result()

print(result.get_counts(my_first_score))
