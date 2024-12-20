# https://github.com/NickyBar/QIP/blob/11747b40beb38d41faa297fb2b53f28c6519c753/qiskit/extensions/standard/ccx.py
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
Toffoli gate. Controlled-Controlled-X.

Author: Andrew Cross
"""
from qiskit import QuantumCircuit
from qiskit import Gate
from qiskit import CompositeGate
from qiskit.extensions.standard import header


class ToffoliGate(Gate):
    """Toffoli gate."""

    def __init__(self, ctl1, ctl2, tgt, circ=None):
        """Create new Toffoli gate."""
        super(ToffoliGate, self).__init__("ccx", [], [ctl1, ctl2, tgt], circ)

    def qasm(self):
        """Return OPENQASM string."""
        ctl1 = self.arg[0]
        ctl2 = self.arg[1]
        tgt = self.arg[2]
        return self._qasmif("ccx %s[%d],%s[%d],%s[%d];" % (ctl1[0].name,
                                                           ctl1[1],
                                                           ctl2[0].name,
                                                           ctl2[1],
                                                           tgt[0].name,
                                                           tgt[1]))

    def inverse(self):
        """Invert this gate."""
        return self  # self-inverse

    def reapply(self, circ):
        """Reapply this gate to corresponding qubits in circ."""
        self._modifiers(circ.ccx(self.arg[0], self.arg[1], self.arg[2]))


def ccx(self, ctl1, ctl2, tgt):
    """Apply Toffoli to circuit."""
    self._check_qubit(ctl1)
    self._check_qubit(ctl2)
    self._check_qubit(tgt)
    self._check_dups([ctl1, ctl2, tgt])
    return self._attach(ToffoliGate(ctl1, ctl2, tgt, self))


QuantumCircuit.ccx = ccx
CompositeGate.ccx = ccx
