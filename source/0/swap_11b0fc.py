# https://github.com/vardhan9/The_Math_of_Intelligence/blob/a432914e1f550c9b41b2fc8d874254168143d2ee/Week10/qiskit-sdk-py-master/qiskit/extensions/standard/swap.py
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
SWAP gate.
"""
from qiskit import QuantumCircuit
from qiskit import Gate
from qiskit import CompositeGate
from qiskit.extensions.standard import header


class SwapGate(Gate):
    """SWAP gate."""

    def __init__(self, ctl, tgt, circ=None):
        """Create new SWAP gate."""
        super(SwapGate, self).__init__("swap", [], [ctl, tgt], circ)

    def qasm(self):
        """Return OPENQASM string."""
        ctl = self.arg[0]
        tgt = self.arg[1]
        return self._qasmif("swap %s[%d],%s[%d];" % (ctl[0].name, ctl[1],
                                                     tgt[0].name, tgt[1]))

    def inverse(self):
        """Invert this gate."""
        return self  # self-inverse

    def reapply(self, circ):
        """Reapply this gate to corresponding qubits in circ."""
        self._modifiers(circ.swap(self.arg[0], self.arg[1]))


def swap(self, ctl, tgt):
    """Apply SWAP from ctl to tgt."""
    self._check_qubit(ctl)
    self._check_qubit(tgt)
    self._check_dups([ctl, tgt])
    return self._attach(SwapGate(ctl, tgt, self))


QuantumCircuit.swap = swap
CompositeGate.swap = swap
