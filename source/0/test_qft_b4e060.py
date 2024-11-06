# https://github.com/wiktor145/QTrator/blob/de8f4ce389bb9b6168096ab615e5b3b8de2b1c11/test_qft.py
from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.circuit.library import QFT


def convert_to_good_qasm(qasm, i):
    gate_definition_line = ""
    qasm_copy = ""
    for line in qasm.split("\n"):
        # print(line[:50])
        if line.startswith("gate gate_QFT"):
            gate_definition_line = line
            break

    # print(gate_definition_line.find("{"))

    qft_inside = gate_definition_line[gate_definition_line.find("{") + 1: gate_definition_line.find("}")]

    for a in range(i - 1, -1, -1):
        qft_inside = qft_inside.replace("q" + str(a), "q[{}]".format(a))

    qft_inside = ";\n".join(qft_inside.split(';'))

    for line in qasm.split("\n"):
        # print(line[:50])
        if line.startswith("gate gate_QFT"):
            pass
        elif line.startswith("gate_QFT"):
            qasm_copy += qft_inside + "\n"
        else:
            qasm_copy += line + "\n"

    return qasm_copy  # tutaj rozwinac bramki, zeby je zrobic inline


for i in range(2, 62):
    circuit = QFT(i)
    circuit.add_register(ClassicalRegister(i))
    circuit.barrier()
    for a in range(i):
        circuit.measure(a, a)
    # print(circuit.qasm())
    qasm = convert_to_good_qasm(circuit.qasm(), i)
    # print(qasm)

    with open("qft/qft{}.qasm".format(i), "w") as f:
        f.write(qasm)
