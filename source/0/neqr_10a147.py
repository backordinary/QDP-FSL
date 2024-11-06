# https://github.com/MatthieuSarkis/Ising-QIP/blob/109af50110681262cade85f93d7e879b6697f18a/src/data_factory/neqr.py
# -*- coding: utf-8 -*-
#
# Written by Matthieu Sarkis (https://github.com/MatthieuSarkis).
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


import numpy as np
from qiskit import QuantumCircuit, Aer, IBMQ, transpile, QuantumRegister, ClassicalRegister
from qiskit.tools.jupyter import *
from qiskit.utils import QuantumInstance
from qiskit.result.counts import Counts
from math import ceil, log2
from typing import Dict, List


def binary_formatting(
    digit: int,
    n_bits: int,
    reverse: bool = False,
) -> str:
    r"""
    Args:
        digit (int): number whose binary representation we are computing
        n_bits (int): number of bits used in the binary representation (to handle left trailing zeros)
        reverse (bool): optionally return the binary representation of digit in reverse order (for qiskit convention)
    Returns:
        (str): binary representation of digit
    """

    binary = format(digit, '0{}b'.format(n_bits))

    if reverse:
        binary = binary[::-1]

    return binary

def image_to_circuit(
    image: np.ndarray,
    measure: bool = False,
) -> QuantumCircuit:

    image_size = image.shape[0]
    side_qubits = ceil(log2(image_size))
    image_qubits = 2 * side_qubits
    total_qubits = image_qubits + 1

    image_register = QuantumRegister(image_qubits, 'position')
    spin_register = QuantumRegister(1,'spin')

    if measure:
        classical_register = ClassicalRegister(image_qubits + 1, 'classical register')
        qc = QuantumCircuit(spin_register, image_register, classical_register)

    else:
        qc = QuantumCircuit(spin_register, image_register)

    qc.i(0)
    for i in range(1, image_register.size + 1):
        qc.h(i)

    qc.barrier()

    for i in range(image_size):
        for j in range(image_size):

            if image[i, j] == 1:

                binarized_i = binary_formatting(digit=i, n_bits=side_qubits, reverse=False)
                binarized_j = binary_formatting(digit=j, n_bits=side_qubits, reverse=False)

                flip_idx = []

                for ii in range(side_qubits):
                    if binarized_i[ii] == '1':
                        flip_idx.append(ii+1)
                        qc.x(ii+1)

                for jj in range(side_qubits):
                    if binarized_j[jj] == '1':
                        flip_idx.append(jj+side_qubits+1)
                        qc.x(jj+side_qubits+1)

                qc.mcx(list(range(1, image_qubits+1)), 0, mode='noancilla')

                for q in flip_idx:
                    qc.x(q)

                qc.barrier()

    if measure:
        qc.measure(range(total_qubits), range(total_qubits))

    return qc


def counts_to_statevector(
    counts: Dict[str, int],
    num_qubits: int,
) -> np.ndarray:

    statevector = np.zeros(shape=(2**num_qubits,))
    for key, value in counts.items():
        statevector[int(key, 2)] = value
    statevector = statevector / np.linalg.norm(statevector)
    return statevector

def statevector_to_densitymatrix(
    statevector: np.ndarray
) -> np.ndarray:
    r"""
    Args:
        statevector (np.ndarray): statevector corresponding to a pure state
    Return:
        (np.ndarray): density matrix corresponding to the pure state statevector
    """

    density_matrix = np.outer(statevector, statevector)
    return density_matrix

def images_to_quantumstate(
    image_batch: np.ndarray,
    from_counts: bool = False,
    to_real: bool = False,
    output_densitymatrices: bool = False
) -> List[np.ndarray]:
    r"""
    Args:
        image_batch (np.ndarray): batch of images whose quantum representation we are computing
        from_counts (np.ndarray): whether or not to reconstruct the output statevector statistically from counts or not
        to_real (bool): whether or not to cast the complex components of statevector to real ones
        output_densitymatrices (bool): whether to output the density matrices of the statevectors
    Return:
        (List[np.ndarray]): batch of density matrix representation of the images
    """

    backend = Aer.get_backend('statevector_simulator')
    bounds = []
    num_qubits = 2 * ceil(log2(image_batch.shape[-1])) + 1

    for i in range(image_batch.shape[0]):
        q = image_to_circuit(image=image_batch[i], measure=True if from_counts else False)
        bounds.append(q)

    qc = transpile(bounds, backend)

    if from_counts:
        qi = QuantumInstance(backend, seed_transpiler=42, seed_simulator=42, shots=2048)
        result = qi.execute(circuits=qc, had_transpiled=True)
        counts = result.get_counts()

        # Handle the case of a batch containing a single image
        if type(counts) == Counts:
            counts = [counts]

        if to_real:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                statevector = [counts_to_statevector(counts[ind], num_qubits).astype('float32') for ind in range(len(counts))]
        else:
            statevector = [counts_to_statevector(counts[ind], num_qubits) for ind in range(len(counts))]

    else:
        qi = QuantumInstance(backend, seed_transpiler=42, seed_simulator=42)
        result = qi.execute(circuits=qc, had_transpiled=True)

        if to_real:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                statevector = [result.get_statevector(i).astype('float32') for ind in range(len(bounds))]
        else:
            statevector = [result.get_statevector(i) for i in range(len(bounds))]

    statevector = np.stack(statevector)

    if output_densitymatrices:
        densitymatrices = np.einsum('ab,ac->abc', statevector, statevector)
        return densitymatrices

    else:
        return statevector