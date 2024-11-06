# https://github.com/lukasszz/qiskit-exp/blob/ce14d53735870e7b6ace352629eb4049e9cd6740/bbdays/cnot3.py
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram


def init_state(state, qc, qr):
    for i, s in enumerate(state):
        if '1' == s:
            qc.x(qr[i])


def circ_a(qc, qr):
    qc.cx(qr[1], qr[2])
    qc.cx(qr[0], qr[1])
    qc.cx(qr[1], qr[2])
    qc.cx(qr[0], qr[1])
    qc.cx(qr[1], qr[2])


def circ_b(qc, qr):
    qc.cx(qr[0], qr[1])
    qc.cx(qr[1], qr[2])
    qc.cx(qr[1], qr[2])
    qc.cx(qr[0], qr[1])


def circ_c(qc, qr):
    qc.cx(qr[1], qr[2])
    qc.cx(qr[0], qr[1])
    qc.cx(qr[1], qr[2])
    qc.cx(qr[0], qr[1])


if __name__ == '__main__':

    backend = Aer.get_backend('qasm_simulator')
    QR = QuantumRegister(3)
    CR = ClassicalRegister(3)

    states = ['100', '101', '001', '000', '011']
    for s in states:
        QC = QuantumCircuit(QR, CR)
        init_state(s, QC, QR)
        circ_a(QC, QR)
        # circ_b(QC, QR)
        #circ_c(QC, QR) #SWAP
        # circ_d(QC, QR)
        # print(QC.draw(output='text'))
        QC.measure(QR, CR)

        job = execute(QC, backend=backend, shots=1024)
        st = job.result().get_counts()
        print(s)
        print(st)
        # plot_histogram(st)
