# https://github.com/niefermar/CuanticaProgramacion/blob/cf066149b4bd769673e83fd774792e9965e5dbc0/qiskit/extensions/standard/cu3.py
# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
controlled-u3 gate.
"""
from qiskit import CompositeGate
from qiskit import Gate
from qiskit import QuantumCircuit
from qiskit._instructionset import InstructionSet
from qiskit._quantumregister import QuantumRegister
from qiskit.extensions.standard import header  # pylint: disable=unused-import


class Cu3Gate(Gate):
    """controlled-u3 gate."""

    def __init__(self, theta, phi, lam, ctl, tgt, circ=None):
        """Create new cu3 gate."""
        super().__init__("cu3", [theta, phi, lam], [ctl, tgt], circ)

    def qasm(self):
        """Return OPENQASM string."""
        ctl = self.arg[0]
        tgt = self.arg[1]
        theta = self.param[0]
        phi = self.param[1]
        lam = self.param[2]
        return self._qasmif("cu3(%s,%s,%s) %s[%d],%s[%d];" % (theta, phi, lam,
                                                              ctl[0].name, ctl[1],
                                                              tgt[0].name, tgt[1]))

    def inverse(self):
        """Invert this gate."""
        self.param[0] = -self.param[0]
        phi = self.param[1]
        self.param[1] = -self.param[2]
        self.param[2] = -phi
        return self

    def reapply(self, circ):
        """Reapply this gate to corresponding qubits in circ."""
        self._modifiers(circ.cu3(self.param[0], self.param[1],
                                 self.param[2], self.arg[0], self.arg[1]))


def cu3(self, theta, phi, lam, ctl, tgt):
    """Apply cu3 from ctl to tgt with angle theta, phi, lam."""
    if isinstance(ctl, QuantumRegister) and \
       isinstance(tgt, QuantumRegister) and len(ctl) == len(tgt):
        instructions = InstructionSet()
        for i in range(ctl.size):
            instructions.add(self.cu3(theta, phi, lam, (ctl, i), (tgt, i)))
        return instructions

    if isinstance(ctl, QuantumRegister):
        instructions = InstructionSet()
        for j in range(ctl.size):
            instructions.add(self.cu3(theta, phi, lam, (ctl, j), tgt))
        return instructions

    if isinstance(tgt, QuantumRegister):
        instructions = InstructionSet()
        for j in range(tgt.size):
            instructions.add(self.cu3(theta, phi, lam, ctl, (tgt, j)))
        return instructions

    self._check_qubit(ctl)
    self._check_qubit(tgt)
    self._check_dups([ctl, tgt])
    return self._attach(Cu3Gate(theta, phi, lam, ctl, tgt, self))


QuantumCircuit.cu3 = cu3
CompositeGate.cu3 = cu3
