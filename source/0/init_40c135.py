# https://github.com/carstenblank/dc-qiskit-stochastics/blob/83813e48a97cace886713b3001a3ce0e60247b88/dc_qiskit_stochastics/qiskit_util/__init__.py
# Copyright 2018-2022 Carsten Blank
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
import qiskit
from qiskit.providers.ibmq import ibmqbackend

from dc_quantum_scheduling.qiskit.qiskit_provider import provider
from qiskit_aer.backends.aerbackend import AerBackend


def ibmqx2() -> 'ibmqbackend.IBMQBackend':
    return provider().get_backend('ibmqx2')

def ibmq_ourense() -> 'ibmqbackend.IBMQBackend':
    return provider().get_backend('ibmq_ourense')


def ibmq_vigo() -> 'ibmqbackend.IBMQBackend':
    return provider().get_backend('ibmq_vigo')


def qasm_simulator() -> 'AerBackend':
    return qiskit.Aer.get_backend('qasm_simulator')


def ibmq_simulator() -> 'ibmqbackend.IBMQBackend':
    return provider().get_backend('ibmq_qasm_simulator')
