# https://github.com/FilipeChagasDev/register-by-constant-qft-adder/blob/f05c6998f00b5f06b4119ffced5dc4f31c0b9113/qiskit_reg_by_const_qft_adder.py
#By Filipe Chagas
#2022

import qiskit
from math import *
from qiskit_quantum_fourier_transform import qft
from numeric_systems import *

def reg_by_const_fourier_basis_adder(n: int, c: int) -> qiskit.circuit.Gate:
    """
    Register-by-constant addition gate in Fourier basis.
    Get a gate to perform an addition of a constant $c$ to an integer register in Fourier basis.
    No ancillary qubits needed.
    
    [see https://doi.org/10.48550/arXiv.2207.05309]

    :param n: Number of target qubits.
    :type n: int
    :param c: Constant to add.
    :type c: int
    :return: $U_{\phi(+)}(c)$ gate.
    :rtype: qiskit.circuit.Gate
    """
    assert n > 0

    my_circuit = qiskit.QuantumCircuit(n, name=f'$U_{{\\phi(+)}}({c})$')

    for i in range(n):
        theta = c * (pi / (2**(n-i-1)))
        my_circuit.rz(theta, i)

    return my_circuit.to_gate()

def reg_by_const_qft_adder(n: int, c: int) -> qiskit.circuit.Gate:
    """
    Register-by-constant QFT addition gate.
    Get a gate to perform an addition of a constant $c$ to a integer register.
    No ancillary qubits needed.

    [see https://doi.org/10.48550/arXiv.2207.05309]

    :param n: Number of target qubits.
    :type n: int
    :param c: Constant to add.
    :type c: int
    :return: $U_{+}(c)$ gate.
    :rtype: qiskit.circuit.Gate
    """
    assert n > 0

    my_circuit = qiskit.QuantumCircuit(n, name=f'$U_{{+}}({c})$')

    my_qft = qft(n)
    my_circuit.append(my_qft, list(range(n)))
    my_circuit.append(reg_by_const_fourier_basis_adder(n, c), list(range(n)))
    my_circuit.append(my_qft.inverse(), list(range(n)))

    return my_circuit.to_gate()