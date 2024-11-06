# https://github.com/svenpruefer/quantumcomputing/blob/e082b6b829ccabdf1c9c64b5cc310ba8feaad2d5/qsoc/costs/costs.py
# -*- coding: utf-8 -*-

# This file is part of qsoc.
#
# Copyright (c) 2020 by DLR.

from typing import OrderedDict
from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Unroller


def _calc_u3_cx_gates(qc: QuantumCircuit) -> OrderedDict[str, int]:
    pass_ = Unroller(['u3', 'cx'])
    pm = PassManager(pass_)
    new_circuit = pm.run(qc)
    return new_circuit.count_ops()


def calc_total_costs(qc: QuantumCircuit) -> int:
    """
    Calculates cost of a QuantumCircuit according to the formula::

       cost = #{u3 gates} + 10 * #{cx gates}

    The number of gates are calculated by using qiskit's built-in unroller.

    :param qc: QuantumCircuit whose costs shall be calculated.
    :return: The total cost of `qc`.
    """
    counts = _calc_u3_cx_gates(qc)
    total = 0
    if 'u3' in counts.keys():
        total = total + counts['u3']
    if 'cx' in counts.keys():
        total = total + 10 * counts['cx']
    return total
