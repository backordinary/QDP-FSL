# https://github.com/averyanalex/quantpiler/blob/71ffd8823d58102d7899b9bf4b4a6156934861bb/tests/test_qram.py
from quantpiler import qram

from qiskit import QuantumRegister, ClassicalRegister
from qiskit.circuit import QuantumCircuit

from quantpiler.utils import uint_to_bits, execute_qc_once


def check_qram(qram, addr, data):
    aqr = QuantumRegister(2)
    dqr = QuantumRegister(3)
    rqr = ClassicalRegister(3)

    qc = QuantumCircuit(aqr, dqr, rqr)

    k = uint_to_bits(addr, 2)
    v = uint_to_bits(data, 3)
    if k[0]:
        qc.x(aqr[0])
    if k[1]:
        qc.x(aqr[1])
    qc.barrier()

    qc.compose(qram, qubits=[0, 1, 2, 3, 4], inplace=True)
    qc.barrier()
    qc.measure(dqr, rqr)

    bits = execute_qc_once(qc, measure=False)

    v_str = ""
    for v_b in v:
        v_str = v_str + str(v_b)

    assert bits[::-1] == v_str


def test_qram_dict():
    values = {0: 1, 1: 3, 2: 6, 3: 7}
    ram = qram.new_qram(2, 3, values)

    for addr in values:
        data = values[addr]

        check_qram(ram, addr, data)


def test_qram_list():
    values = [1, 3, 6, 7]
    ram = qram.new_qram(2, 3, values)

    for addr in range(len(values)):
        data = values[addr]

        check_qram(ram, addr, data)
