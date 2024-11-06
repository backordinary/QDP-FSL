# https://github.com/talm95/QuantumComputing/blob/accf5574da5ecf8da7c233db928caf2ae74c9d4a/ArithmeticOperations/addition.py
from math import pi

from qiskit import QuantumCircuit

from ShorsAlgorithm.QuantumFourierTransform import qft


def addition_3n_plus_1(a, b):
    single_circuit_size = max(a.num_qubits, b.num_qubits)
    a_plus_b = QuantumCircuit(single_circuit_size * 3 + 1, single_circuit_size + 1)
    a_plus_b.append(a, range(single_circuit_size))
    a_plus_b.append(b, range(single_circuit_size, 2 * single_circuit_size))

    for qubit in range(0, single_circuit_size):
        if qubit > 0:
            a_plus_b.ccx(qubit, qubit + 2 * single_circuit_size, qubit + 2 * single_circuit_size + 1)
            a_plus_b.ccx(qubit + single_circuit_size, qubit + 2*single_circuit_size, qubit + 2*single_circuit_size + 1)

        a_plus_b.ccx(qubit, qubit + single_circuit_size, qubit + 2*single_circuit_size + 1)
        a_plus_b.cx(qubit, qubit + 2 * single_circuit_size)
        a_plus_b.cx(qubit + single_circuit_size, qubit + 2 * single_circuit_size)
        a_plus_b.barrier()

    for register in range(single_circuit_size + 1):
        a_plus_b.measure(register + single_circuit_size * 2, register)

    return a_plus_b


def qft_addition(a, b):
    return plus_minus_with_qft(a, b, True)


def qft_subtraction(a, b):
    return plus_minus_with_qft(b, a, False)


def plus_minus_with_qft(a, b, is_plus):
    single_circuit_size = max(a.num_qubits, b.num_qubits) + 1

    # create a_plus_minus_b circuit
    a_plus_minus_b = QuantumCircuit(2 * single_circuit_size, single_circuit_size)
    a_plus_minus_b.append(a, range(single_circuit_size - 1))
    a_plus_minus_b.append(b, range(single_circuit_size, 2 * single_circuit_size - 1))

    # use qft circuit on b value
    qft_circuit = QuantumCircuit(single_circuit_size, name='QFT')
    qft(qft_circuit, single_circuit_size)
    a_plus_minus_b.append(qft_circuit, range(single_circuit_size, 2 * single_circuit_size))

    # perform operation of a and b in the fourier space
    phase = pi
    if not is_plus:
        phase = -pi

    for power in range(0, single_circuit_size):
        for controlled in range(single_circuit_size - 1 - power, -1, -1):
            a_plus_minus_b.cp(phase / (2 ** power), controlled, single_circuit_size + controlled + power)

    # reverse qft
    a_plus_minus_b.append(qft_circuit.inverse(), range(single_circuit_size, 2 * single_circuit_size))

    # perform measure
    for register in range(single_circuit_size):
        a_plus_minus_b.measure(register + single_circuit_size, register)

    return a_plus_minus_b
