# https://github.com/vardhan9/The_Math_of_Intelligence/blob/a432914e1f550c9b41b2fc8d874254168143d2ee/Week10/qiskit-sdk-py-master/qiskit/extensions/standard/y.py
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
Pauli Y (bit-phase-flip) gate.
"""
from qiskit import QuantumRegister
from qiskit import QuantumCircuit
from qiskit import Gate
from qiskit import CompositeGate
from qiskit import InstructionSet
from qiskit.extensions.standard import header


class YGate(Gate):
    """Pauli Y (bit-phase-flip) gate."""

    def __init__(self, qubit, circ=None):
        """Create new Y gate."""
        super(YGate, self).__init__("y", [], [qubit], circ)

    def qasm(self):
        """Return OPENQASM string."""
        qubit = self.arg[0]
        return self._qasmif("y %s[%d];" % (qubit[0].name, qubit[1]))

    def inverse(self):
        """Invert this gate."""
        return self  # self-inverse

    def reapply(self, circuit):
        """Reapply this gate to corresponding qubits in circ."""
        self._modifiers(circuit.y(self.arg[0]))


def y(self, quantum_register):
    """Apply Y to q."""
    if isinstance(quantum_register, QuantumRegister):
        intructions = InstructionSet()
        for register in range(quantum_register.size):
            intructions.add(self.y((quantum_register, register)))
        return intructions
    else:
        self._check_qubit(quantum_register)
        return self._attach(YGate(quantum_register, self))


QuantumCircuit.y = y
CompositeGate.y = y
