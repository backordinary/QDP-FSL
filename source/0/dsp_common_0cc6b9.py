# https://github.com/carstenblank/dc-qiskit-stochastics/blob/83813e48a97cace886713b3001a3ce0e60247b88/dc_qiskit_stochastics/dsp_common.py
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
import logging

import numpy as np
import qiskit
from dc_qiskit_algorithms import UniformRotationGate
from qiskit.circuit import Parameter
from qiskit.circuit.library import U1Gate
from scipy import sparse

LOG = logging.getLogger(__name__)


def apply_initial(value: float, scaling_factor: Parameter) -> qiskit.QuantumCircuit:
    """
    This function initializes the circuit using Proposition 1 of the paper:
    First we need a Hadamard and then we rotate by the value * scaling factor * 2
    with the R_z rotation.
    :param value: The initial value
    :param scaling_factor: The scaling factor to be used when computing the characteristic function
    :return: The initial quantum circuit with the data system only
    """
    qc = qiskit.QuantumCircuit(name='initial_rotation')
    qreg_data = qiskit.QuantumRegister(1, 'data')
    qc.add_register(qreg_data)
    # BY Proposition 1 we need to start in a superposition state
    qc.h(qreg_data)
    # Then the initial rotation
    # qc.append(RZGate(2*scaling_factor * value), qreg_data)
    qc.u1(scaling_factor * value, qreg_data)
    return qc


def apply_level(level: int, realizations: np.array, scaling_factor: Parameter, with_debug_circuit: bool = False, **kwargs) -> qiskit.QuantumCircuit:
    k, = realizations.shape
    qubits_k = int(np.ceil(np.log2(k)))
    qc = qiskit.QuantumCircuit(name=f'level_{level}')
    qreg_index = qiskit.QuantumRegister(qubits_k, f'level_{level}')
    qreg_data = qiskit.QuantumRegister(1, 'data')
    qc.add_register(qreg_index)
    qc.add_register(qreg_data)
    alpha = sparse.dok_matrix([realizations]).transpose()

    if with_debug_circuit:
        LOG.debug(f"Will add a controlled rotations with u1({scaling_factor} * {realizations})")
        for (i, j), angle in alpha.items():
            qc.append(
                U1Gate(1.0 * scaling_factor * angle).control(num_ctrl_qubits=qubits_k, ctrl_state=int(i)),
                list(qreg_index) + list(qreg_data)
            )
    else:
        LOG.debug(f"Will add a uniform rotation gate with u1({scaling_factor} * {realizations})")
        qc.append(UniformRotationGate(lambda theta: U1Gate(scaling_factor * theta), alpha), list(qreg_index) + list(qreg_data))

    return qc


def apply_level_two_realizations(level: int, realizations: np.array, scaling_factor: Parameter, **kwargs) -> qiskit.QuantumCircuit:
    k, = realizations.shape
    assert k == 2
    qubits_k = int(np.ceil(np.log2(k)))
    qc = qiskit.QuantumCircuit(name=f'level_{level}')
    qreg_index = qiskit.QuantumRegister(qubits_k, f'level_{level}')
    qreg_data = qiskit.QuantumRegister(1, 'data')
    qc.add_register(qreg_index)
    qc.add_register(qreg_data)
    x_l1 = realizations[0]
    x_l2 = realizations[1]
    qc.cu1(theta=scaling_factor * x_l2, control_qubit=qreg_index, target_qubit=qreg_data)
    qc.x(qreg_index)
    qc.cu1(theta=scaling_factor * x_l1, control_qubit=qreg_index, target_qubit=qreg_data)
    qc.x(qreg_index)
    return qc


def x_measurement() -> qiskit.QuantumCircuit:
    qc = qiskit.QuantumCircuit(name='initial_rotation')
    qreg_data = qiskit.QuantumRegister(1, 'data')
    creg_data = qiskit.ClassicalRegister(1, 'output')
    qc.add_register(qreg_data)
    qc.add_register(creg_data)
    qc.h(qreg_data)
    qc.measure(qreg_data, creg_data)
    return qc


def y_measurement() -> qiskit.QuantumCircuit:
    qc = qiskit.QuantumCircuit(name='initial_rotation')
    qreg_data = qiskit.QuantumRegister(1, 'data')
    creg_data = qiskit.ClassicalRegister(1, 'output')
    qc.add_register(qreg_data)
    qc.add_register(creg_data)
    qc.z(qreg_data)
    qc.s(qreg_data)
    qc.h(qreg_data)
    qc.measure(qreg_data, creg_data)
    return qc
