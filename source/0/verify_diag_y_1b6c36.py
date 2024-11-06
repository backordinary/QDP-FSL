# https://github.com/IBM/Simultaneous-diagonalization/blob/385545401395a2e07f109441db4751a5dcf8f0a4/verify_diag_y.py
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

y = np.asarray([[0,-1j],[1j,0]])

angle = 0.3

(Ev,Eb) = np.linalg.eig(y)
expy1 = np.dot(Eb,np.dot(np.diag(np.exp(1j*angle*Ev)), Eb.T.conj()))

circuit = QuantumCircuit(2)
circuit.s(0)
circuit.h(0)
circuit.x(0)
circuit.cx(0,1)
circuit.rz(-angle,1)
circuit.x(1)
circuit.rz(angle,1)
circuit.x(1)
circuit.cx(0,1)
circuit.x(0)
circuit.h(0)
circuit.sdg(0)

backend = Aer.get_backend('unitary_simulator')
U = execute(circuit, backend).result().get_unitary()
expy2 = U[:2,:2]
print("Error = %s" % np.linalg.norm(expy1 - expy2,'fro'))
