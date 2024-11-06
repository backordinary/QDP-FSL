# https://github.com/Talkal13/Quantum/blob/ccda55776da0a3f5bd212a8566f0a1e367061a6f/Lab/measure.py

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.aqua.circuits import StateVectorCircuit
import numpy as np
from math import pi, acos

def execute_qc(qc):
    backend = Aer.get_backend("qasm_simulator")
    job = execute(qc, backend, shots=1000)
    return (job.result().get_statevector(), job.result().get_counts())


qb = QuantumRegister(1)
cb = ClassicalRegister(1)
q = QuantumCircuit(qb, cb)
p = 0.85
lamb = acos((p - 0.5)*2)
q.h(qb)
q.rz(pi/4, qb)
q.h(qb)

q.measure(qb, cb)

result = execute_qc(q)
print(result)



