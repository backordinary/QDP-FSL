# https://github.com/BryceFuller/quantum-mobile-backend/blob/5211f07813e581a00d6cee9e1b02dea55d7d7f50/qiskit/extensions/standard/ubase.py
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
Element of SU(2).
"""
from qiskit import Gate
from qiskit.extensions._extensionerror import ExtensionError
from qiskit import QuantumCircuit
from qiskit import CompositeGate


class UBase(Gate):
    """Element of SU(2)."""

    def __init__(self, param, qubit, circ=None):
        """Create new reset instruction."""
        if len(param) != 3:
            raise ExtensionError("expected 3 parameters")
        super(UBase, self).__init__("U", param, [qubit], circ)

    def qasm(self):
        """Return OPENQASM string."""
        theta = self.param[0]
        phi = self.param[1]
        lamb = self.param[2]
        qubit = self.arg[0]
        return self._qasmif("U(%.15f,%.15f,%.15f) %s[%d];" % (theta, phi,
                                                              lamb, qubit[0].name, qubit[1]))

    def inverse(self):
        """Invert this gate.

        U(theta,phi,lambda)^dagger = U(-theta,-lambda,-phi)
        """
        self.param[0] = -self.param[0]
        phi = self.param[1]
        self.param[1] = -self.param[2]
        self.param[2] = -phi
        return self

    def reapply(self, circ):
        """Reapply this gate to corresponding qubits in circ."""
        self._modifiers(circ.ubase(self.arg[0]))


def u_base(self, tpl, q):
    """Apply U to q."""
    self._check_qubit(q)
    return self._attach(UBase(tpl, q, self))


QuantumCircuit.u_base = u_base
CompositeGate.u_base = u_base
