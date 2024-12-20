# https://github.com/CharlesKocher/simulating-mss-qc-on-quantum-computers/blob/c1d1d30189320ae37868fee5c00fc3aa5a5aaa91/qiskit-env/variational_forms/ryspecialqho.py
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
#
# Modified from QISKit ACQUA
import numpy as np
from qiskit import QuantumRegister, QuantumCircuit

from qiskit_acqua.utils.variational_forms import VariationalForm


class VarFormRYSpecial(VariationalForm):
    """Single Y rotation."""

    RYSpecialQHO_CONFIGURATION = {
        'name': 'RYSpecialQHO',
        'description': 'RYSpecialQHO Variational Form',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'ryspecialqho_schema',
            'type': 'object',
            'properties': {
                'depth': {
                    'type': 'integer',
                    'default': 1,
                    'minimum': 1
                },
                'entanglement': {
                    'type': 'string',
                    'default': 'full',
                    'oneOf': [
                        {'enum': ['full', 'linear']}
                    ]
                },
                'entangler_map': {
                    'type': ['object', 'null'],
                    'default': None
                }
            },
            'additionalProperties': False
        }
    }

    def __init__(self, configuration=None):
        super().__init__(configuration or self.RYSpecialQHO_CONFIGURATION.copy())
        self._num_qubits = 0
        self._depth = 0
        self._entangler_map = None
        self._initial_state = None

    def init_args(self, num_qubits, depth, entangler_map=None,
                  entanglement='full', initial_state=None):
        """
        Args:
            num_qubits (int) : number of qubits
            depth (int) : number of rotation layers, but will be set to 1
            entangler_map (dict) : dictionary of entangling gates, in the format
                                    { source : [list of targets] },
                                    or None for full entanglement.
            entanglement (str): 'full' or 'linear'
            initial_state (InitialState): an initial state object
        """
        depth = 1
        self._num_parameters = 1
        self._bounds = [(-np.pi, np.pi)] * self._num_parameters
        self._num_qubits = num_qubits
        self._depth = depth
        if entangler_map is None:
            entangler_map = dict()
            #self._entangler_map = VariationalForm.get_entangler_map(entanglement, num_qubits)
        #else:
            #self._entangler_map = VariationalForm.validate_entangler_map(entangler_map, num_qubits)
        self._initial_state = initial_state

    def construct_circuit(self, parameters):
        """
        Construct the variational form, given its parameters.

        Args:
            parameters (numpy.ndarray): circuit parameters.

        Returns:
            QuantumCircuit: a quantum circuit with given `parameters`

        Raises:
            ValueError: the number of parameters is incorrect.
        """
        if len(parameters) != self._num_parameters:
            raise ValueError('The number of parameters has to be {}'.format(self._num_parameters))

        q = QuantumRegister(self._num_qubits, name='q')
        if self._initial_state is not None:
            circuit = self._initial_state.construct_circuit('circuit', q)
        else:
            circuit = QuantumCircuit(q)

        param_idx = 0
        circuit.ry(parameters[0],q[0])
        circuit.barrier(q)

        return circuit
