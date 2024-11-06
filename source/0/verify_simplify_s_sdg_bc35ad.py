# https://github.com/IBM/Simultaneous-diagonalization/blob/385545401395a2e07f109441db4751a5dcf8f0a4/verify_simplify_s_sdg.py
# Copyright 2022 IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This file is part of the code to reproduce the results in the paper:
# E. van den Berg and Kristan Temme, "Circuit optimization of Hamiltonian
# simulation by simultaneous diagonalization of Pauli clusters," Quantum 4,
# p. 322, 2020. https://doi.org/10.22331/q-2020-09-12-322

from qiskit import *
import numpy as np

circuit = QuantumCircuit(1)
circuit.s(0)
circuit.sdg(0)
print(circuit.draw())

circuit_opt = compiler.transpile(circuit, optimization_level=2)
print(circuit_opt.draw())

# Evaluate unitaries
backend = Aer.get_backend('unitary_simulator')

circuit = QuantumCircuit(1)
circuit.s(0)
S = execute(circuit, backend).result().get_unitary()

circuit =QuantumCircuit(1)
circuit.sdg(0)
Sdg = execute(circuit, backend).result().get_unitary()


# Show the unitaries corresponding to S and Sdg, along with
# their product, which should be identity.
print(S)
print(Sdg)
print(np.dot(S,Sdg))
print(np.dot(Sdg,S))
