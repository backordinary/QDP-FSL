# https://github.com/rum-yasuhiro/palloq/blob/8559c8c1827dc38e782f6a07f8bfc50659eb983b/test/compiler/test_dynamic_multiqc_compose.py
# test for dynamic_multiqc_compose()

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.test.mock import FakeMelbourne, FakeParis

from palloq.compiler.dynamic_multiqc_compose import dynamic_multiqc_compose

"""This test is written as pytest style"""


def test_small_queue_grid():
    # prepare mock backend
    backend = FakeMelbourne()
    coupling_map = FakeMelbourne().configuration().coupling_map
    basis_gates = FakeMelbourne().configuration().basis_gates

    # prepare qcs
    qcs = []

    # prepare test qc1
    qr1 = QuantumRegister(3, "q1")
    cr1 = ClassicalRegister(3)
    qc1 = QuantumCircuit(qr1, cr1)
    # weight 2 on (0, 1) and weight 1 on (1, 2)
    qc1.cx(qr1[0], qr1[1])
    qc1.cx(qr1[1], qr1[0])
    qc1.cx(qr1[1], qr1[2])
    qc1.measure(qr1, cr1)
    qcs.append(qc1)

    # prepare test qc2
    qr2 = QuantumRegister(3, "q2")
    cr2 = ClassicalRegister(3)
    qc2 = QuantumCircuit(qr2, cr2)
    # weight 1 on (0, 1) and weight 2 on (1, 2)
    qc2.cx(qr2[0], qr2[1])
    qc2.cx(qr2[1], qr2[2])
    qc2.cx(qr2[2], qr2[1])
    qc2.measure(qr2, cr2)
    qcs.append(qc2)

    transpiled_qcs = dynamic_multiqc_compose(
        queued_qc=qcs,
        backend=FakeMelbourne(),
        coupling_map=coupling_map,
        basis_gates=basis_gates,
        routing_method="sabre",
    )

    for _qc in transpiled_qcs:
        print()
        print(_qc)


def test_small_queue_falconr4():
    # prepare mock backend
    backend = FakeParis()
    coupling_map = FakeParis().configuration().coupling_map
    basis_gates = FakeParis().configuration().basis_gates

    # prepare qcs
    qcs = []

    # prepare test qc1
    qr1 = QuantumRegister(3, "q1")
    cr1 = ClassicalRegister(3)
    qc1 = QuantumCircuit(qr1, cr1)
    # weight 2 on (0, 1) and weight 1 on (1, 2)
    qc1.cx(qr1[0], qr1[1])
    qc1.cx(qr1[1], qr1[0])
    qc1.cx(qr1[1], qr1[2])
    qc1.measure(qr1, cr1)
    qcs.append(qc1)

    # prepare test qc2
    qr2 = QuantumRegister(3, "q2")
    cr2 = ClassicalRegister(3)
    qc2 = QuantumCircuit(qr2, cr2)
    # weight 1 on (0, 1) and weight 2 on (1, 2)
    qc2.cx(qr2[0], qr2[1])
    qc2.cx(qr2[1], qr2[2])
    qc2.cx(qr2[2], qr2[0])
    qc2.cx(qr2[0], qr2[1])
    qc2.cx(qr2[1], qr2[2])
    qc2.cx(qr2[2], qr2[0])
    qc2.measure(qr2, cr2)
    qcs.append(qc2)

    transpiled_qcs = dynamic_multiqc_compose(
        queued_qc=qcs,
        backend=FakeParis(),
        coupling_map=coupling_map,
        basis_gates=basis_gates,
        routing_method="sabre",
        scheduling_method="alap",
    )

    for _qc in transpiled_qcs:
        print()
        print(_qc)


def test_qasmbench_s_falconr4(small_circuits):
    # prepare mock backend
    backend = FakeParis()
    coupling_map = FakeParis().configuration().coupling_map
    basis_gates = FakeParis().configuration().basis_gates

    # prepare qcs
    qcs = []
    qcs = small_circuits[:2]

    transpiled_qcs = dynamic_multiqc_compose(
        queued_qc=qcs,
        backend=FakeParis(),
        coupling_map=coupling_map,
        basis_gates=basis_gates,
        routing_method="sabre",
        scheduling_method="alap",
    )

    for _qc in transpiled_qcs:
        print()
        print(_qc)
