# https://github.com/lukasszz/qiskit-exp/blob/ce14d53735870e7b6ace352629eb4049e9cd6740/bbdays/ry.py
from math import pi

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram


def ry(angle):
    global circ, job, st
    circ = QuantumCircuit(q, b)
    circ.ry(angle, q[0])
    # circ.h(q)
    circ.measure(q[0], b[0])
    print(circ)
    job = execute(circ, backend=backend, shots=1024)
    st = job.result().get_counts()
    print(st)


def ryh(angle):
    global circ, job, st
    circ = QuantumCircuit(q, b)
    circ.ry(angle, q[0])
    circ.h(q)
    circ.measure(q[0], b[0])
    print(circ)
    job = execute(circ, backend=backend, shots=1024)
    st = job.result().get_counts()
    print(st)


if __name__ == '__main__':
    backend = Aer.get_backend('qasm_simulator')
    q = QuantumRegister(1)
    b = ClassicalRegister(1)
    circ = QuantumCircuit(q, b)

    ry(pi / 2)

    ry(pi / 4)

    ryh(pi / 4)
