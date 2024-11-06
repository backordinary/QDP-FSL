# https://github.com/svenpruefer/quantumcomputing/blob/e082b6b829ccabdf1c9c64b5cc310ba8feaad2d5/qsoc/circuits/adder.py
# -*- coding: utf-8 -*-

# This file is part of qsoc.
#
# Copyright (c) 2020 by DLR.

from qiskit import QuantumCircuit
from qiskit.circuit import Qubit
from qsoc.circuits.classic import (add_xor, add_and)


def add_half_adder(qc: QuantumCircuit, a: Qubit, b: Qubit, sum: Qubit, carry: Qubit) -> None:
    """
    Add a half adder to a QuantumCircuit.
    The circuit will look like the following::

    | TODO

    :param qc: Underlying QuantumCircuit.
    :param a: First qubit to be added.
    :param b: Second qubit to be added.
    :param sum: a XOR b will be XORed to initial state of this qubit, i.e this will be a XOR b, if initial state was |0>.
    :param carry: a AND b will be XORed to initial state of this qubit, i.e this will be a AND b, if initial state was |0>.
    """
    add_and(qc, a, b, carry)
    add_xor(qc, a, b, sum)


def add_full_adder_7(qc: QuantumCircuit, a: Qubit, b: Qubit, c: Qubit, aux_1: Qubit, aux_2: Qubit, sum: Qubit,
                     carry: Qubit) -> None:
    """
    Add a full adder using two ancillary qubits to a QuantumCircuit.
    The circuit will look like the following::

    | TODO

    :param qc: Underlying QuantumCircuit.
    :param a: First qubit to be added, not modified.
    :param b: Second qubit to be added, not modified.
    :param c: Third qubit to be added, not modified, usually the carry of a formerly applied full adder.
    :param aux_1: Ancillary qubit. Result will be initial state XORed with (a XOR b), i.e. this will be (a XOR b) if initial state was |0>.
    :param aux_2: Ancillary qubit. Result will be initial state XORed with (a AND b), i.e. this will be (a AND b) if initial state was |0>.
    :param sum: (a XOR b XOR c) will be XORed to initial state of this qubit, i.e this will be (a XOR b XOR c), if initial state was |0>.
    :param carry: Carry of a + b + c will be XORed to initial state of this qubit, i.e this will be the carry of a + b +c, if initial state was |0>.
    """
    add_half_adder(qc, a, b, aux_1, aux_2)
    add_half_adder(qc, c, aux_1, sum, carry)
    qc.cx(aux_2, carry)


def add_full_adder_6(qc: QuantumCircuit, a: Qubit, b: Qubit, c: Qubit, aux: Qubit, sum: Qubit, carry: Qubit) -> None:
    """
    Add a full adder using one ancillary qubit to a QuantumCircuit.
    The circuit will look like the following::

    | TODO

    :param qc: Underlying QuantumCircuit.
    :param a: First qubit to be added, not modified.
    :param b: Second qubit to be added, not modified.
    :param c: Third qubit to be added, not modified, usually the carry of a formerly applied full adder.
    :param aux: Ancillary qubit. Result will be initial state XORed with (a XOR b), i.e. this will be (a XOR b) if initial state was |0>.
    :param sum: (a XOR b XOR c) will be XORed to initial state of this qubit, i.e this will be (a XOR b XOR c), if initial state was |0>.
    :param carry: Carry of a + b + c will be XORed to initial state of this qubit, i.e this will be the carry of a + b +c, if initial state was |0>.
    """
    add_half_adder(qc, a, b, aux, carry)
    add_half_adder(qc, c, aux, sum, carry)


def add_full_adder_5(qc: QuantumCircuit, a: Qubit, b: Qubit, c: Qubit, sum: Qubit, carry: Qubit) -> None:
    """
    Add a full adder to a QuantumCircuit.
    The circuit will look like the following::

    | TODO

    :param qc: Underlying QuantumCircuit.
    :param a: First qubit to be added, not modified.
    :param b: Second qubit to be added, not modified.
    :param c: Third qubit to be added, not modified, usually the carry of a formerly applied full adder.
    :param sum: (a XOR b XOR c) will be XORed to initial state of this qubit, i.e this will be (a XOR b XOR c), if initial state was |0>.
    :param carry: Carry of a + b + c will be XORed to initial state of this qubit, i.e this will be the carry of a + b +c, if initial state was |0>.
    """
    add_half_adder(qc, a, b, sum, carry)
    qc.ccx(c, sum, carry)
    qc.cx(c, sum)


def add_full_adder_5_reverse(qc: QuantumCircuit, a: Qubit, b: Qubit, c: Qubit, sum: Qubit, carry: Qubit) -> None:
    """
    Add a reverse full adder to a QuantumCircuit, to be used e.g. in Grover when one needs to reset the sum and carry qubits.
    """
    qc.cx(c, sum)
    qc.ccx(c, sum, carry)
    # A half adder is the reverse of itsel because all operations commute
    add_half_adder(qc, a, b, sum, carry)
