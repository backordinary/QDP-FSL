# https://github.com/niefermar/CuanticaProgramacion/blob/cf066149b4bd769673e83fd774792e9965e5dbc0/qiskit/extensions/simulator/noise.py
# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Simulator command to toggle noise off or on.
"""
from qiskit import Instruction
from qiskit import QuantumCircuit
from qiskit import CompositeGate
from qiskit import QuantumRegister
from qiskit.extensions._extensionerror import ExtensionError
from qiskit.extensions.standard import header  # pylint: disable=unused-import


class Noise(Instruction):
    """Simulator noise operation."""

    def __init__(self, switch, qubits, circ):
        """Create new noise instruction."""
        super().__init__("noise", [switch], list(qubits), circ)

    def inverse(self):
        """Special case. Return self."""
        return self

    def qasm(self):
        """Return OPENQASM string."""
        string = "noise(%d) " % self.param[0]
        for j in range(len(self.arg)):
            if len(self.arg[j]) == 1:
                string += "%s" % self.arg[j].name
            else:
                string += "%s[%d]" % (self.arg[j][0].name, self.arg[j][1])
            if j != len(self.arg) - 1:
                string += ","
        string += ";"
        return string

    def reapply(self, circ):
        """Reapply this instruction to corresponding qubits in circ."""
        self._modifiers(circ.noise(self.param[0]))


def noise(self, switch):
    """Turn noise on/off in simulator.
    Works on all qubits, and prevents reordering (like barrier).

    Args:
        switch (int): turn noise on (1) or off (0)

    Returns:
        QuantumCircuit: with attached command

    Raises:
        ExtensionError: malformed command
    """
    tuples = []
    if isinstance(self, QuantumCircuit):
        for register in self.regs.values():
            if isinstance(register, QuantumRegister):
                tuples.append(register)
    if not tuples:
        raise ExtensionError("no qubits for noise")
    if switch is None:
        raise ExtensionError("no noise switch passed")
    qubits = []
    for tuple_element in tuples:
        if isinstance(tuple_element, QuantumRegister):
            for j in range(tuple_element.size):
                self._check_qubit((tuple_element, j))
                qubits.append((tuple_element, j))
        else:
            self._check_qubit(tuple_element)
            qubits.append(tuple_element)
    self._check_dups(qubits)
    return self._attach(Noise(switch, qubits, self))


# Add to QuantumCircuit and CompositeGate classes
QuantumCircuit.noise = noise
CompositeGate.noise = noise
