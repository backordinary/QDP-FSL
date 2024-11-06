# https://github.com/Talkal13/Quantum/blob/ccda55776da0a3f5bd212a8566f0a1e367061a6f/QCP/aritmetic/boolean.py
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library.standard_gates import XGate

def qnot():

    q = QuantumRegister(1)
    qc = QuantumCircuit(q)
    qc.x(q)

    return qc

def xor():
    q = QuantumRegister(2)
    qc = QuantumCircuit(q)
    qc.cx(q[0], q[1])

    return qc

def qand():

    q = QuantumRegister(2)
    a = QuantumRegister(1)
    qc = QuantumCircuit(q, a)
    qc.ccx(q[0], q[1], a)

    return qc

def qor(n=2):
    q = QuantumRegister(n)
    a = QuantumRegister(1)
    qc = QuantumCircuit(q, a, name="Or(%i)" % n)
    
    qc.x(q)
    qc.append(XGate().control(n), q[:] + [a])
    qc.x(q)
    qc.x(a)

    return qc
