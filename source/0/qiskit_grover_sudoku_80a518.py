# https://github.com/kohpai/qc-python/blob/8addf3a971826edf17eb5c59ee9457a3a84345f6/qiskit_grover_sudoku.py
from qiskit import Aer, transpile
from qiskit.circuit import AncillaRegister, ClassicalRegister, Gate, QuantumCircuit, QuantumRegister
from qiskit.circuit.library import HGate, ZGate


def xor():
    qr = QuantumRegister(3)
    qc = QuantumCircuit(qr)

    qc.cx(qr[0], qr[2])
    qc.cx(qr[1], qr[2])
    return qc


def check_rows_columns():
    qr = QuantumRegister(4, 'in')
    ar = AncillaRegister(4, 'tmp')
    qc = QuantumCircuit(qr, ar)

    qc.append(xor(), [qr[0], qr[1], ar[0]])
    qc.append(xor(), [qr[2], qr[3], ar[1]])
    qc.append(xor(), [qr[0], qr[2], ar[2]])
    qc.append(xor(), [qr[1], qr[3], ar[3]])
    return qc


def sudoku_oracle():
    qr = QuantumRegister(4)
    ar = AncillaRegister(4)
    out = QuantumRegister(1)
    qc = QuantumCircuit(qr, ar, out)

    qc.append(check_rows_columns(), list(qr) + list(ar))
    qc.mcx(ar, out)
    qc.append(check_rows_columns().inverse(), list(qr) + list(ar))
    return qc


def reflect_zero(nqubits: int):
    qr = QuantumRegister(nqubits)
    qc = QuantumCircuit(qr)

    qc.x(qr)
    qc.append(ZGate().control(nqubits - 1), qr)
    qc.x(qr)
    return qc


def reflect_initial(init_gate: Gate, nqubits: int):
    qr = QuantumRegister(nqubits)
    qc = QuantumCircuit(qr)

    qc.append(init_gate.inverse(), [qr])
    qc.append(reflect_zero(nqubits), qr)
    qc.append(init_gate, [qr])
    return qc


def create_grover_circuit():
    qr = QuantumRegister(4, 'in')
    ar = AncillaRegister(4, 'anc')
    pkb = QuantumRegister(1, 'pkb')
    output = ClassicalRegister(4, 'out')

    qc = QuantumCircuit(qr, ar, pkb, output)

    qc.h(qr)
    qc.x(pkb)
    qc.h(pkb)
    qc.append(sudoku_oracle(), list(qr) + list(ar) + list(pkb))
    qc.append(reflect_initial(HGate(), 4), qr)
    qc.append(sudoku_oracle(), list(qr) + list(ar) + list(pkb))
    qc.append(reflect_initial(HGate(), 4), qr)

    qc.measure(qr, output)
    return qc


circuit = create_grover_circuit()
backend = Aer.get_backend('aer_simulator')
results = backend.run(transpile(circuit, backend), shots=512).result()
counts = results.get_counts()
print(counts["0110"])
print(counts["1001"])
