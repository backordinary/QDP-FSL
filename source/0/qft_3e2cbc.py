# https://github.com/carstenblank/dc-qiskit-algorithms/blob/f197d9005b19ce162b0962aa72423f95bc53aa88/dc_qiskit_algorithms/Qft.py
# Copyright 2018 Carsten Blank
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
import math
from typing import Tuple, List, Union

from qiskit import QuantumRegister, QuantumCircuit
from qiskit.circuit import Gate, Instruction, Qubit, Clbit
from qiskit.extensions import HGate, CU1Gate


def get_theta(k):
    # type: (int) -> float
    r"""
    As the conditional rotations are defined by a parameter k we have a
    convenience function for this:

    .. math::

        \theta = \pm \frac{2\pi}{2^{(|k|)}}

    :param k: the integer for the angle
    :return: the angle
    """
    sign = -1 if k < 0 else 1
    lam = sign * 2 * math.pi * 2**(-abs(k))
    return lam


class QuantumFourierTransformGate(Gate):

    def __init__(self, num_qubits):
        super().__init__("qft", num_qubits=num_qubits, params=[])

    def _define(self):
        rule = []  # type: List[Tuple[Gate, List[Qubit], List[Clbit]]]
        qreg = QuantumRegister(self.num_qubits, "qreg")
        qc = QuantumCircuit(qreg, name=self.name)
        q_list = list(qreg)

        unused = q_list.copy()
        for qr in q_list:
            rule.append((HGate(), [qr], []))
            k = 2
            unused.remove(qr)
            for qj in unused:
                rule.append((CU1Gate(get_theta(k)), [qj, qr], []))
                k = k + 1

        qc._data = rule.copy()
        self.definition = qc

    def inverse(self):
        return super().inverse()


def qft(self, q):
    # type: (QuantumCircuit, Union[List[Qubit], QuantumRegister]) -> Instruction
    """
    Applies the Quantum Fourier Transform to q

    :param self: the circuit to which the qft is applied
    :param q: the quantum register or list of quantum register/index tuples
    :return: the circuit with applied qft
    """
    return self.append(QuantumFourierTransformGate(len(q)), [q])


def qft_dg(self, q):
    # type: (QuantumCircuit, Union[List[Qubit], QuantumRegister]) -> Instruction
    """
        Applies the inverse Quantum Fourier Transform to q

        :param self: the circuit to which the qft_dag is applied
        :param q: the quantum register or list of quantum register/index tuples
        :return: the circuit with applied qft_dag
        """
    return qft(self, q).inverse()
