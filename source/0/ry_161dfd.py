# https://github.com/BryceFuller/quantum-mobile-backend/blob/5211f07813e581a00d6cee9e1b02dea55d7d7f50/qiskit/extensions/standard/ry.py
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
Rotation around the y-axis.
"""
from qiskit import QuantumRegister
from qiskit import QuantumCircuit
from qiskit import Gate
from qiskit import InstructionSet
from qiskit import CompositeGate
from qiskit.extensions.standard import header


class RYGate(Gate):
    """rotation around the y-axis."""

    def __init__(self, theta, qubit, circ=None):
        """Create new ry single qubit gate."""
        super(RYGate, self).__init__("ry", [theta], [qubit], circ)

    def qasm(self):
        """Return OPENQASM string."""
        qubit = self.arg[0]
        theta = self.param[0]
        return self._qasmif("ry(%.15f) %s[%d];" % (theta, qubit[0].name,
                                                   qubit[1]))

    def inverse(self):
        """Invert this gate.

        ry(theta)^dagger = ry(-theta)
        """
        self.param[0] = -self.param[0]
        return self

    def reapply(self, circ):
        """Reapply this gate to corresponding qubits in circ."""
        self._modifiers(circ.ry(self.param[0], self.arg[0]))


def ry(self, theta, q):
    """Apply ry to q."""
    if isinstance(q, QuantumRegister):
        gs = InstructionSet()
        for j in range(q.sz):
            gs.add(self.ry(theta, (q, j)))
        return gs
    else:
        self._check_qubit(q)
        return self._attach(RYGate(theta, q, self))


QuantumCircuit.ry = ry
CompositeGate.ry = ry
