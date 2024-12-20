# https://github.com/BryceFuller/quantum-mobile-backend/blob/5211f07813e581a00d6cee9e1b02dea55d7d7f50/qiskit/extensions/standard/gatestools.py
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

from qiskit import QuantumRegister
from qiskit import InstructionSet


def attach_gate(element, quantum_register, gate, gate_class):
    if isinstance(quantum_register, QuantumRegister):
        gs = InstructionSet()
        for register in range(quantum_register.size):
            gs.add(gate)
        return gs
    else:
        element._check_qubit(quantum_register)
        return element._attach(gate_class)
