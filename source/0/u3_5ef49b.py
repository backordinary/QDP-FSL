# https://github.com/indisoluble/qasm-simulator-cpp-debian-pkg/blob/5fb81719defacf8731b96c74d934d7bf9191afa8/qiskit-sdk-py-0.5.1/qiskit/extensions/standard/u3.py
# -*- coding: utf-8 -*-
# pylint: disable=invalid-name

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
Two-pulse single-qubit gate.
"""
from qiskit import CompositeGate
from qiskit import Gate
from qiskit import InstructionSet
from qiskit import QuantumCircuit
from qiskit import QuantumRegister
from qiskit.extensions.standard import header  # pylint: disable=unused-import


class U3Gate(Gate):
    """Two-pulse single-qubit gate."""

    def __init__(self, theta, phi, lam, qubit, circ=None):
        """Create new two-pulse single qubit gate."""
        super().__init__("u3", [theta, phi, lam], [qubit], circ)

    def qasm(self):
        """Return OPENQASM string."""
        qubit = self.arg[0]
        theta = self.param[0]
        phi = self.param[1]
        lam = self.param[2]
        return self._qasmif("u3(%s,%s,%s) %s[%d];" % (theta, phi, lam,
                                                      qubit[0].name,
                                                      qubit[1]))

    def inverse(self):
        """Invert this gate.

        u3(theta, phi, lamb)^dagger = u3(-theta, -lam, -phi)
        """
        self.param[0] = -self.param[0]
        phi = self.param[1]
        self.param[1] = -self.param[2]
        self.param[2] = -phi
        return self

    def reapply(self, circ):
        """Reapply this gate to corresponding qubits in circ."""
        self._modifiers(circ.u3(self.param[0], self.param[1], self.param[2],
                                self.arg[0]))


def u3(self, theta, phi, lam, q):
    """Apply u3 to q."""
    if isinstance(q, QuantumRegister):
        instructions = InstructionSet()
        for j in range(q.size):
            instructions.add(self.u3(theta, phi, lam, (q, j)))
        return instructions

    self._check_qubit(q)
    return self._attach(U3Gate(theta, phi, lam, q, self))


QuantumCircuit.u3 = u3
CompositeGate.u3 = u3
