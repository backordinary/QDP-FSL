# https://github.com/rum-yasuhiro/experiments_crosstalk_multitasking/blob/5d92c40dcfc68fab7922e806ab75e898bf3bda81/test/test_insert_delay.py
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.test.mock.backends import FakeAthens

from experiments.waiting_duration import InsertDelay

def test_before_opeartion(): 
    qr = QuantumRegister(3)
    cr = ClassicalRegister(3)
    qc = QuantumCircuit(qr, cr)
    qc.h(qr)
    qc.cx(qr[0], qr[1])
    qc.cx(qr[1], qr[2])

    print(qc)

    delayed_qc = InsertDelay(1000, qc).before_operation()
    print(delayed_qc)


def test_before_measurement(): 
    qr = QuantumRegister(3)
    cr = ClassicalRegister(3)
    qc = QuantumCircuit(qr, cr)
    qc.h(qr)
    qc.cx(qr[0], qr[1])
    qc.cx(qr[1], qr[2])

    print(qc)

    delayed_qc = InsertDelay(1000, qc).before_measurement()
    print(delayed_qc)


if __name__ == "__main__":
    test_before_opeartion()
    test_before_measurement()