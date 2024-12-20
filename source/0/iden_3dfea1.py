# https://github.com/NickyBar/QIP/blob/11747b40beb38d41faa297fb2b53f28c6519c753/qiskit/extensions/standard/iden.py
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
Identity gate.

Author: Andrew Cross
"""
from qiskit import QuantumCircuit
from qiskit import QuantumRegister
from qiskit import Gate
from qiskit import CompositeGate
from qiskit import InstructionSet
from qiskit.extensions.standard import header


class IdGate(Gate):
    """Identity gate."""

    def __init__(self, qubit, circ=None):
        """Create new Identity gate."""
        super(IdGate, self).__init__("id", [], [qubit], circ)

    def qasm(self):
        """Return OPENQASM string."""
        qubit = self.arg[0]
        return self._qasmif("id %s[%d];" % (qubit[0].name, qubit[1]))

    def inverse(self):
        """Invert this gate."""
        return self  # self-inverse

    def reapply(self, circ):
        """Reapply this gate to corresponding qubits in circ."""
        self._modifiers(circ.id(self.arg[0]))


def iden(self, q):
    """Apply Identity to q."""
    if isinstance(q, QuantumRegister):
        gs = InstructionSet()
        for j in range(q.size):
            gs.add(self.iden((q, j)))
        return gs
    else:
        self._check_qubit(q)
        return self._attach(IdGate(q, self))


QuantumCircuit.iden = iden
CompositeGate.iden = iden
