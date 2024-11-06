# https://github.com/HK-ilohas/quantum_simulation/blob/717c21fc7551f50ab4603f6e51e948b2858d6ab7/shor.py
from fractions import Fraction

from qiskit import *

from arithmetic_circuit import *


def shor_algorithm(num_qubits: int, a: int, n: int) -> int:
    # check args
    assert gcd(a, n) == 1
    assert a.bit_length() <= num_qubits
    assert n.bit_length() <= num_qubits

    n_counts = num_qubits * 2

    # define qubits
    qr_x = QuantumRegister(n_counts, "x")  # target
    qr_y = QuantumRegister(num_qubits, "y")  # 1
    qr_a = QuantumRegister(num_qubits, "a")
    qr_b = QuantumRegister(num_qubits, "b")
    qr_co = QuantumRegister(1, "co")  # carry out
    qr_n = QuantumRegister(num_qubits, "n")
    qr_z = QuantumRegister(1, "z")
    qr_list = [qr_x, qr_y, qr_a, qr_b, qr_co, qr_n, qr_z]
    # define circuit
    qc = QuantumCircuit(*qr_list)

    init_register(qc, qr_n, n)
    init_register(qc, qr_y, 1)

    for i in range(n_counts):
        qc.h(qr_x[i])

    for i in range(n_counts):
        qc.append(c_Ua(num_qubits, pow(a, 2**i, n), n), qr_x[i:i+1] + qr_y[:] + qr_a[:] + qr_b[:] + qr_co[:] + qr_n[:] + qr_z[:])

    # measure
    cr = ClassicalRegister(n_counts)
    qc.add_register(cr)

    for i in range(n_counts):
        qc.measure(qr_x[i], i)

    # Transpile for simulator
    simulator = Aer.get_backend("aer_simulator")
    compiled_circuit = transpile(qc, simulator)
    # Run and get counts
    job = simulator.run(compiled_circuit)
    result = job.result()
    counts = result.get_counts(compiled_circuit)

    # find r using fractions
    for output in counts:
        decimal = int(output, 2)
        phase = decimal/(2**n_counts)
        frac = Fraction(phase).limit_denominator(n)
        r = frac.denominator
        if pow(a, r, n) == 1:
            return r
    else:
        return False
