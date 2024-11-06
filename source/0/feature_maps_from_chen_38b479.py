# https://github.com/shaqiyu/QC_HEP_Hcc/blob/07c8d8adae4a040596bef9607da5522011bf996c/QHEP/Qhep_Modules_0820/feature_maps_from_Chen.py
# -*- coding: utf-8 -*-

# Copyright 2018 IBM.
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
This module contains the definition of a base class for
feature map. Several types of commonly used approaches.
"""

from collections import OrderedDict
import copy
import itertools
import logging

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Pauli
from qiskit.qasm import pi
from sympy.core.numbers import NaN, Float

from qiskit.aqua.components.feature_maps import PauliExpansion, FeatureMap, self_product

logger = logging.getLogger(__name__)

class FeatureMaps(PauliExpansion):
    """
    Mapping data with the second order expansion followed by entangling gates.
    Refer to https://arxiv.org/pdf/1804.11326.pdf for details.
    """

    CONFIGURATION = {
        'name': 'MultiVariableMap',
        'description': 'Second order expansion for feature map',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'Second_Order_Expansion_schema',
            'type': 'object',
            'properties': {
                'depth': {
                    'type': 'integer',
                    'default': 2,
                    'minimum': 1
                },
                'entangler_map': {
                    'type': ['array', 'null'],
                    'default': None
                },
                'entanglement': {
                    'type': 'string',
                    'default': 'full',
                    'oneOf': [
                        {'enum': ['full', 'linear']}
                    ]
                }
            },
            'additionalProperties': False
        }
    }

    def __init__(self, num_qubits, depth=2, degree=1, entangler_map=None,
                 entanglement='full', data_map_func=self_product):
        """Constructor.
        Args:
            num_qubits (int): number of qubits
            depth (int): the number of repeated circuits
            entangler_map (list[list]): describe the connectivity of qubits, each list describes
                                        [source, target], or None for full entanglement.
                                        Note that the order is the list is the order of
                                        applying the two-qubit gate.
            entanglement (str): ['full', 'linear'], generate the qubit connectivitiy by predefined
                                topology
            data_map_func (Callable): a mapping function for data x
        """
        #self.validate(locals())
        super().__init__(num_qubits, depth, entangler_map, entanglement,
                         data_map_func=data_map_func)
        self._support_parameterized_circuit = False
        self._degree = degree

    def _build_circuit_template(self):
        return True

    def construct_circuit(self, x, qr=None, inverse=False):
        #logger.info("8 qubits 26 variable feature map is used")
        """
        Construct the second order expansion based on given data.
        Args:
            x (numpy.ndarray): 1-D to-be-transformed data.
            qr (QauntumRegister): the QuantumRegister object for the circuit, if None,
                                  generate new registers with name q.
            inverse (bool): whether or not inverse the circuit
        Returns:
            QuantumCircuit: a quantum circuit transform data x.
        """

        if qr is None:
            qr = QuantumRegister(self._num_qubits, name='q')

        qc = QuantumCircuit(qr)

        entangler_map = []
        for i in range(0, self._num_qubits - 1, 2):
            entangler_map.append([i, i + 1])
        for i in range(0, self._num_qubits - 2, 2):
            entangler_map.append([i + 1, i + 2])

        for d in range(self._depth):
            for i in range(self._num_qubits):
                qc.h(qr[i])
                qc.rz(x[i], qr[i])
                if self._degree>0:
                   qc.ry(x[i]**self._degree,qr[i])
            for src, targ in entangler_map:
                qc.cx(qr[src], qr[targ])
                qc.rz((((x[src] + x[targ])/2)**self._degree), qr[targ])
                qc.cx(qr[src], qr[targ])
        if inverse:
            qc = qc.inverse()
        return qc
