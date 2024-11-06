# https://github.com/lukasszz/qiskit-exp/blob/ce14d53735870e7b6ace352629eb4049e9cd6740/bbdays/1plus1.py
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram


def init_state(state, qc, qr):
    if '00' == state:
        pass
    elif '01' == state:
        qc.x(qr[0])
    elif '10' == state:
        qc.x(qr[1])
    elif '11' == state:
        qc.x(qr[0])
        qc.x(qr[1])


def circ_a(qc, qr):
    qc.h(qr[1])
    qc.cx(qr[0], qr[1])
    qc.h(qr[1])


def circ_b(qc, qr):
    qc.h(qr[0])
    qc.cx(qr[0], qr[1])
    qc.h(qr[0])


def circ_c(qc, qr):
    qc.h(qr[0])
    qc.h(qr[1])
    qc.cx(qr[0], qr[1])
    qc.h(qr[0])
    qc.h(qr[1])


if __name__ == '__main__':

    backend = Aer.get_backend('qasm_simulator')
    q = QuantumRegister(4)
    b = ClassicalRegister(2)
    circ = QuantumCircuit(q, b)

    circ.x(q[0])
    circ.x(q[1])
    circ.barrier()
    circ.ccx(q[0], q[1], q[3])
    circ.cx(q[0], q[2])
    circ.cx(q[1], q[2])
    circ.barrier()
    circ.measure(q[2], b[0])
    circ.measure(q[3], b[1])
    print(circ)

    job = execute(circ, backend=backend, shots=1024)
    st = job.result().get_counts()
    print(st)

