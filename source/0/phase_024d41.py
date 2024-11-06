# https://github.com/Talkal13/Quantum/blob/ccda55776da0a3f5bd212a8566f0a1e367061a6f/Lab/phase.py
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

# Subroutine to change a phase

def phase_change(theta, n, Uf):
    x = QuantumRegister(n)
    a = QuantumRegister(1)
    q = QuantumCircuit(x, a, name="phase(" + str(theta) + ")")

    q.h(x)
    q.append(Uf, x[:] + [a[0]])
    q.u1(theta, a)
    q.append(Uf.inverse(), x[:] + [a[0]])
    return q