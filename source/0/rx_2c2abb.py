# https://github.com/mauget/qiskit-sdk-py/blob/a863e2df4e59ae6a74962cfb4fb4693b2af37cfc/qiskit/extensions/standard/rx.py
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
Rotation around the x-axis.
"""
from qiskit import QuantumRegister
from qiskit import QuantumCircuit
from qiskit import Gate
from qiskit import InstructionSet
from qiskit import CompositeGate
from qiskit.extensions.standard import header


class RXGate(Gate):
    """rotation around the x-axis."""

    def __init__(self, theta, qubit, circ=None):
        """Create new rx single qubit gate."""
        super(RXGate, self).__init__("rx", [theta], [qubit], circ)

    def qasm(self):
        """Return OPENQASM string."""
        qubit = self.arg[0]
        theta = self.param[0]
        return self._qasmif("rx(%s) %s[%d];" % (theta, qubit[0].name,
                                                   qubit[1]))

    def inverse(self):
        """Invert this gate.

        rx(theta)^dagger = rx(-theta)
        """
        self.param[0] = -self.param[0]
        return self

    def reapply(self, circ):
        """Reapply this gate to corresponding qubits in circ."""
        self._modifiers(circ.rx(self.param[0], self.arg[0]))


def rx(self, theta, q):
    """Apply rx to q."""
    if isinstance(q, QuantumRegister):
        gs = InstructionSet()
        for j in range(q.sz):
            gs.add(self.rx(theta, (q, j)))
        return gs
    else:
        self._check_qubit(q)
        return self._attach(RXGate(theta, q, self))


QuantumCircuit.rx = rx
CompositeGate.rx = rx
