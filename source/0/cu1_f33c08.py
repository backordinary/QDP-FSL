# https://github.com/vardhan9/The_Math_of_Intelligence/blob/a432914e1f550c9b41b2fc8d874254168143d2ee/Week10/qiskit-sdk-py-master/qiskit/extensions/standard/cu1.py
# -*- coding: utf-8 -*-

# Copyright 2017 IBM RESEARCH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

"""
controlled-u1 gate.
"""
from qiskit import QuantumCircuit
from qiskit import Gate
from qiskit import CompositeGate
from qiskit.extensions.standard import header
from qiskit._quantumregister import QuantumRegister
from qiskit._instructionset import InstructionSet


class Cu1Gate(Gate):
    """controlled-u1 gate."""

    def __init__(self, theta, ctl, tgt, circ=None):
        """Create new cu1 gate."""
        super(Cu1Gate, self).__init__("cu1", [theta], [ctl, tgt], circ)

    def qasm(self):
        """Return OPENQASM string."""
        ctl = self.arg[0]
        tgt = self.arg[1]
        theta = self.param[0]
        return self._qasmif("cu1(%.15f) %s[%d],%s[%d];" % (theta, ctl[0].name, ctl[1],
                                                           tgt[0].name, tgt[1]))

    def inverse(self):
        """Invert this gate."""
        self.param[0] = -self.param[0]
        return self

    def reapply(self, circ):
        """Reapply this gate to corresponding qubits in circ."""
        self._modifiers(circ.cu1(self.param[0], self.arg[0], self.arg[1]))


def cu1(self, theta, ctl, tgt):
    """Apply cu1 from ctl to tgt with angle theta."""
    if isinstance(ctl, QuantumRegister) and \
       isinstance(tgt, QuantumRegister) and len(ctl) == len(tgt):
        # apply cx to qubits between two registers
        instructions = InstructionSet()
        for i in range(ctl.size):
            instructions.add(self.cu1(theta, (ctl, i), (tgt, i)))
        return instructions
    else:
        self._check_qubit(ctl)
        self._check_qubit(tgt)
        self._check_dups([ctl, tgt])
        return self._attach(Cu1Gate(theta, ctl, tgt, self))


QuantumCircuit.cu1 = cu1
CompositeGate.cu1 = cu1
