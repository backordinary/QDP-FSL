# https://github.com/BryceFuller/quantum-mobile-backend/blob/5211f07813e581a00d6cee9e1b02dea55d7d7f50/qiskit/extensions/standard/s.py
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
S=diag(1,i) Clifford phase gate or its inverse.
"""
import math
from qiskit import QuantumRegister
from qiskit import QuantumCircuit
from qiskit import CompositeGate
from qiskit import InstructionSet
from qiskit.extensions.standard import header
from qiskit.extensions.standard import u1


class SGate(CompositeGate):
    """S=diag(1,i) Clifford phase gate or its inverse."""

    def __init__(self, qubit, circ=None):
        """Create new S gate."""
        super(SGate, self).__init__("s", [], [qubit], circ)
        self.u1(math.pi / 2.0, qubit)

    def reapply(self, circ):
        """Reapply this gate to corresponding qubits in circ."""
        self._modifiers(circ.s(self.arg[0]))

    def qasm(self):
        """Return OPENQASM string."""
        qubit = self.data[0].arg[0]
        phi = self.data[0].param[0]
        if phi > 0:
            return self.data[0]._qasmif("s %s[%d];" % (qubit[0].name, qubit[1]))
        else:
            return self.data[0]._qasmif("sdg %s[%d];" % (qubit[0].name, qubit[1]))


def s(self, q):
    """Apply S to q."""
    if isinstance(q, QuantumRegister):
        gs = InstructionSet()
        for j in range(q.size):
            gs.add(self.s((q, j)))
        return gs
    else:
        self._check_qubit(q)
        return self._attach(SGate(q, self))


def sdg(self, q):
    """Apply Sdg to q."""
    return self.s(q).inverse()


QuantumCircuit.s = s
QuantumCircuit.sdg = sdg
CompositeGate.s = s
CompositeGate.sdg = sdg
