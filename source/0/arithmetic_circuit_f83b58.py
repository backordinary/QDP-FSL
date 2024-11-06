# https://github.com/HK-ilohas/quantum_simulation/blob/717c21fc7551f50ab4603f6e51e948b2858d6ab7/arithmetic_circuit.py
from math import gcd

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library.basis_change import QFT

from utils import init_register


def phi_adder(num_qubits: int):
    qr_a = QuantumRegister(num_qubits, name="a")
    qr_b = QuantumRegister(num_qubits, name="b")
    qr_z = QuantumRegister(1, name="cout")
    qr_list = [qr_a, qr_b, qr_z]

    qc = QuantumCircuit(*qr_list, name="phi_adder")

    for i in range(num_qubits):
        for j in range(num_qubits - i):
            lam = np.pi / (2 ** j)
            qc.cp(lam, qr_a[i], qr_b[i + j])

    for i in range(num_qubits):
        lam = np.pi / (2 ** (i + 1))
        qc.cp(lam, qr_a[num_qubits - i - 1], qr_z[0])

    return qc.to_gate()


def phi_cc_adder_modulo(num_qubits: int):
    # define qubits
    qr_c = QuantumRegister(2, "c")  # control
    qr_a = QuantumRegister(num_qubits, "a")
    qr_b = QuantumRegister(num_qubits + 1, "b")  # avoid to overflow
    qr_n = QuantumRegister(num_qubits, "n")
    qr_z = QuantumRegister(1, "z")  # zero
    qr_list = [qr_c, qr_a, qr_b, qr_n, qr_z]

    # define circuit
    qc = QuantumCircuit(*qr_list, name="phi_cc_adder_modulo")

    # a + b
    qc.append(phi_adder(num_qubits).control(2), qr_c[:] + qr_a[:] + qr_b[:])

    # a + b - n
    qc.append(phi_adder(num_qubits).inverse(), qr_n[:] + qr_b[:])
    # check for overflow
    qc.append(QFT(num_qubits+1, do_swaps=False).inverse().to_gate(), qr_b[:])
    qc.cx(qr_b[-1], qr_z[:])
    qc.append(QFT(num_qubits+1, do_swaps=False).to_gate(), qr_b[:])
    # add n if overflow
    qc.append(phi_adder(num_qubits).control(1), qr_z[:] + qr_n[:] + qr_b[:])

    # sub a
    qc.append(phi_adder(num_qubits).control(2).inverse(), qr_c[:] + qr_a[:] + qr_b[:])
    # restore zero
    qc.append(QFT(num_qubits+1, do_swaps=False).inverse().to_gate(), qr_b[:])
    qc.x(qr_b[-1])
    qc.cx(qr_b[-1], qr_z[:])
    qc.x(qr_b[-1])
    qc.append(QFT(num_qubits+1, do_swaps=False).to_gate(), qr_b[:])
    # add a
    qc.append(phi_adder(num_qubits).control(2), qr_c[:] + qr_a[:] + qr_b[:])

    return qc.to_gate()


def cmult_mod(num_qubits: int, a: int, n: int):
    # define qubits
    qr_c = QuantumRegister(1, "c")  # control
    qr_x = QuantumRegister(num_qubits, "x")
    qr_a = QuantumRegister(num_qubits, "a")
    qr_b = QuantumRegister(num_qubits + 1, "b")  # for overflow
    qr_n = QuantumRegister(num_qubits, "n")  # modulo
    qr_z = QuantumRegister(1, "z")  # for adder modulo
    qr_list = [qr_c, qr_x, qr_a, qr_b, qr_n, qr_z]
    # define circuit
    qc = QuantumCircuit(*qr_list, name="cmult_mod")

    qc.append(QFT(num_qubits + 1, do_swaps=False).to_gate(), qr_b[:])

    for i in range(num_qubits):
        a_i = (2 ** i) * a % n
        init_register(qc, qr_a, a_i)
        qr_cc_adder_modulo = qr_c[:] + qr_x[i:i+1] + qr_a[:] + qr_b[:] + qr_n[:] + qr_z[:]
        qc.append(phi_cc_adder_modulo(num_qubits), qr_cc_adder_modulo)
        init_register(qc, qr_a, a_i)

    qc.append(QFT(num_qubits + 1, do_swaps=False).inverse().to_gate(), qr_b[:])

    return qc.to_gate()


def cmult_mod_inv(num_qubits: int, a_inv: int, n: int):
    # define qubits
    qr_c = QuantumRegister(1, "c")  # control
    qr_x = QuantumRegister(num_qubits, "x")
    qr_a = QuantumRegister(num_qubits, "a")  # zeros (don't init)
    qr_b = QuantumRegister(num_qubits + 1, "b")  # for overflow
    qr_n = QuantumRegister(num_qubits, "n")  # modulo
    qr_z = QuantumRegister(1, "z")  # for adder modulo
    qr_list = [qr_c, qr_x, qr_a, qr_b, qr_n, qr_z]
    # define circuit
    qc = QuantumCircuit(*qr_list, name="cmult_mod")

    qc.append(QFT(num_qubits + 1, do_swaps=False).to_gate(), qr_b[:])

    for i in range(num_qubits - 1, -1, -1):
        a_i = (2 ** i) * a_inv % n
        init_register(qc, qr_a, a_i)
        qr_cc_adder_modulo = qr_c[:] + qr_x[i:i+1] + qr_a[:] + qr_b[:] + qr_n[:] + qr_z[:]
        qc.append(phi_cc_adder_modulo(num_qubits).inverse(), qr_cc_adder_modulo)
        init_register(qc, qr_a, a_i)

    qc.append(QFT(num_qubits + 1, do_swaps=False).inverse().to_gate(), qr_b[:])

    return qc.to_gate()


def c_Ua(num_qubits: int, a: int, n: int):
    # define qubits
    qr_c = QuantumRegister(1, "c")  # control
    qr_x = QuantumRegister(num_qubits, "x")
    qr_a = QuantumRegister(num_qubits, "a")  # must be zero
    qr_b = QuantumRegister(num_qubits, "b")
    qr_co = QuantumRegister(1, "co")  # carry out
    qr_n = QuantumRegister(num_qubits, "n")  # modulo
    qr_z = QuantumRegister(1, "z")  # for adder modulo
    qr_list = [qr_c, qr_x, qr_a, qr_b, qr_co, qr_n, qr_z]
    # define circuit
    qc = QuantumCircuit(*qr_list, name="c_Ua")

    assert gcd(a, n) == 1
    a_inv = pow(a, -1, n)

    qc.append(cmult_mod(num_qubits, a, n), qr_c[:] + qr_x[:] + qr_a[:] + qr_b[:] + qr_co[:] + qr_n[:] + qr_z[:])
    # swap x, b
    for i in range(num_qubits):
        qc.cswap(qr_c[:], qr_x[i], qr_b[i])
    qc.append(cmult_mod_inv(num_qubits, a_inv, n), qr_c[:] + qr_x[:] + qr_a[:] + qr_b[:] + qr_co[:] + qr_n[:] + qr_z[:])

    return qc.to_gate()
