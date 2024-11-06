# https://github.com/georgios-ts/qc-mentorship/blob/8e930b6a3ca6c31b5b947dc0ab58f0fe87715c7e/task_3/library.py
from numpy import pi

from qiskit.circuit import Parameter
from qiskit import QuantumCircuit, QuantumRegister

from basis_library import BasisLibrary

from qiskit.circuit.library.standard_gates import (
    HGate,
    ZGate,  XGate,  YGate,
    RZGate, RXGate, RYGate,
    CXGate, CZGate
)


libr = RxzCzLibrary = BasisLibrary()


# H - Gate
q = QuantumRegister(1, 'q')
_def = QuantumCircuit(q, global_phase=pi / 2)

rules = [(RZGate(pi / 2), [q[0]], []),
         (RXGate(pi / 2), [q[0]], []),
         (RZGate(pi / 2), [q[0]], [])]

for inst, qargs, cargs in rules:
    _def.append(inst, qargs, cargs)

libr.add('h', _def)


# Z - Gate
q = QuantumRegister(1, 'q')
_def = QuantumCircuit(q, global_phase=pi / 2)

rules = [(RZGate(pi), [q[0]], [])]

for inst, qargs, cargs in rules:
    _def.append(inst, qargs, cargs)

libr.add('z', _def)


# X - Gate
q = QuantumRegister(1, 'q')
_def = QuantumCircuit(q, global_phase=pi / 2)

rules = [(RXGate(pi), [q[0]], [])]

for inst, qargs, cargs in rules:
    _def.append(inst, qargs, cargs)

libr.add('x', _def)


# Y - Gate
q = QuantumRegister(1, 'q')
_def = QuantumCircuit(q, global_phase=pi / 2)

rules = [(RXGate(pi), [q[0]], []),
         (RZGate(pi), [q[0]], [])]

for inst, qargs, cargs in rules:
    _def.append(inst, qargs, cargs)

libr.add('y', _def)


# Ry - Gate
q = QuantumRegister(1, 'q')
_def = QuantumCircuit(q)

theta = Parameter('theta')

rules = [(RZGate(-pi / 2), [q[0]], []),
         (RXGate(theta),  [q[0]], []),
         (RZGate(pi / 2), [q[0]], [])]

for inst, qargs, cargs in rules:
    _def.append(inst, qargs, cargs)

libr.add('ry', _def)


# CX - Gate
q = QuantumRegister(2, 'q')
_def = QuantumCircuit(q)

rules = [(RZGate(pi / 2), [q[1]], []),
         (RXGate(pi / 2), [q[1]], []),
         (CZGate(), [q[0], q[1]], []),
         (RXGate(-pi / 2), [q[1]], []),
         (RZGate(-pi / 2), [q[1]], [])]

for inst, qargs, cargs in rules:
    _def.append(inst, qargs, cargs)

libr.add('cx', _def)
