# https://github.com/yahja-dev/QuantumLabs/blob/bcde084ed481e333fe5d159b6413bda4e61ead05/ShorsAlgorithm/methods.py
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT


def initialize(circuit, n, m):
    """
    :param circuit: quantum circuit
    :param n: number of qubits
    :param m: target qubits
    :return: void
    """
    circuit.h(range(n))
    circuit.x(n + m - 1)
    circuit.draw(output='mpl', filename='illustrations/init.png')


def apowerxmod15(a, x):
    """
    :param a: current prime number before 15
    :param x: exponent
    :return: c_U (custom quantum gate)
    """
    if a not in [2,7,8,11,13]:
        raise ValueError("'a' must be 2,7,8,11 or 13")
    U = QuantumCircuit(4)
    for iteration in range(x):
        if a in [2,13]:
            U.swap(0,1)
            U.swap(1,2)
            U.swap(2,3)
        if a in [7,8]:
            U.swap(2,3)
            U.swap(1,2)
            U.swap(0,1)
        if a == 11:
            U.swap(1,3)
            U.swap(0,2)
        if a in [7,11,13]:
            for q in range(4):
                U.x(q)
    U = U.to_gate()
    U.name = "%i^%i mod 15" % (a, x)
    c_U = U.control()
    return c_U


def modular_exponentiation(circuit, n, m, a):
    """
    :param circuit: quantum circuit
    :param n: number of qubits
    :param m: target qubits
    :param a: prime number before 15
    :return: void
    """
    for x in range(n):
        exponent = 2 ** x
        circuit.append(apowerxmod15(a, exponent),
                             [x] + list(range(n, n + m)))
    circuit.draw(output='mpl', filename='illustrations/modexp.png')


def iqft(circuit, measurements):
    """
    :param circuit: quantum circuit
    :param measurements: qubits to be measured
    :return: void
    """
    circuit.append(QFT(len(measurements), do_swaps=False).inverse(), measurements)
    circuit.draw(output='mpl', filename='illustrations/ifqt.png')
