# https://github.com/MatthieuSarkis/Ising-QIP/blob/648c30b407e8c5f3e53a69fd91e613f125d4f728/src/kernel_ridge_regression/kernels/quantum_kernel.py
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

r"""
Implementation of the Quantum Kernel Ridge Regression.
"""

import numpy as np
from typing import Optional
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.result.counts import Counts
from qiskit.utils.mitigation import complete_meas_cal, CompleteMeasFitter
from qiskit.result import Result
from math import ceil, log2
import multiprocess as mp
import itertools

from src.kernel_ridge_regression.abstract_kernels.qiskit_kernel import QiskitKernel
from src.kernel_ridge_regression.abstract_kernels.kernel_ridge_regression import KernelRidgeRegression


class Quantum_Kernel(KernelRidgeRegression, QiskitKernel):
    r"""Class implementing the quantum kernel. The kernel can either be linear or exponential
    depending on whether a value of gamma has been provided to the constructor or not.
    """

    def __init__(
        self,
        use_ancilla: bool = False,
        parallelize: bool = True,
        *args,
        **kwargs,
    ) -> None:
        r"""Constructor for the quantum kernel class.
        Args:
            use_ancilla (bool): whether or not to use ancilla qubits in the compilation of X-gates controlled
            by a large number of qubits.
            parallelize (bool): whether or not to compute the gram matrix entries in parallel using multiple CPUs.
        """

        super(Quantum_Kernel, self).__init__(*args, **kwargs)
        self.name = 'quantum_kernel'

        self.num_qubits: Optional[int] = None
        self.use_ancilla = use_ancilla
        self.parallelize = parallelize

    def _distances_squared(
        self,
        X1: np.ndarray,
        X2: np.ndarray,
    ) -> np.ndarray:
        r"""This method computes the distance_squared matrix in \exp(- \gamma * distance_squared)
        for the quantum kernel, namely ||\rho(x_1) - \rho(x_2)||_F^2 = 2 (1-|<psi_1|psi_2>|^2).
        Args:
            X1 (np.ndarray): First batch of 2d images.
            X2 (np.ndarray): Second batch of 2d images.
        Returns:
            (np.ndarray): Distance squared matrix associated to the two batches of data X1 and X2.
        """

        if self.parallelize:
            overlap_squared = self._overlap_squared_parallel(X1=X1, X2=X2)

        else:
            overlap_squared = self._overlap_squared(X1=X1, X2=X2)

        return 2 * (1 - overlap_squared)

    def _overlap_squared_parallel(
        self,
        X1: np.ndarray,
        X2: np.ndarray,
    ) -> np.ndarray:
        r""" Compute the matrix of the modulus squared of the overlaps |<psi_1|psi_2>|^2.
        Args:
            X1 (np.ndarray): First batch of 2d images.
            X2 (np.ndarray): Second batch of 2d images.
        Returns:
            (np.ndarray): Squared overlap matrix associated to the two batches of data X1 and X2.
        """

        N1 = X1.shape[0]
        N2 = X2.shape[0]

        input = ((i,j) for i, j in itertools.product(range(N1), range(N2)))
        overlap_squared = np.zeros(shape=(N1*N2,))
        pool = mp.Pool(mp.cpu_count()-2)
        result = pool.starmap(lambda i, j: self._overlap_squared(X1=X1[i:i+1], X2=X2[j:j+1]), input)
        for i, r in enumerate(result):
            overlap_squared[i] = r
        overlap_squared.shape = (N1, N2)
        pool.close()
        pool.join()

        return overlap_squared

    def _overlap_squared(
        self,
        X1: np.ndarray,
        X2: np.ndarray,
    ) -> np.ndarray:
        r""" Compute the matrix of the modulus squared of the overlaps |<psi_1|psi_2>|^2.
        Args:
            X1 (np.ndarray): First batch of 2d images.
            X2 (np.ndarray): Second batch of 2d images.
        Returns:
            (np.ndarray): Squared overlap matrix associated to the two batches of data X1 and X2.
        """

        num_qubits = 2 * ceil(log2(X1.shape[-1])) + 1

        bounds1 = []
        for i in range(X1.shape[0]):
            q = self._image_to_circuit(image=X1[i])
            bounds1.append(q)

        bounds2 = []
        for i in range(X2.shape[0]):
            q = self._image_to_circuit(image=X2[i])
            bounds2.append(q)

        # To estimate |<psi_i|psi_j>|^2, glue the Hermitian conjugate of
        # circuit i to circuit j.
        circuits = []
        for c1 in bounds1:
            for c2 in bounds2:
                c = c2.compose(c1.inverse())
                if self._backend_type == 'IBMQ' or self.use_ancilla:
                    c = c.measure_all(inplace=False)
                circuits.append(c)

        qc = transpile(circuits, self._backend)

        # In case one is not using a simulator of requires
        # an ancilla qubit for the implementation of the controlled X-gates
        # it is necessary to perform measurements
        if (self._backend_type == 'IBMQ') or self.use_ancilla:

            # If needed, quantum error mitigation goes here

            result = self.qi.execute(circuits=qc, had_transpiled=True)
            #from qiskit.tools import job_monitor
            #job_monitor(result)
            counts = result.get_counts()

            # Handle the case of a batch containing a single image
            if type(counts) == Counts:
                counts = [counts]

            # Remove the bit coming from measuring the ancilla qubit
            # (recall the convention of qiskit concerning the qubits ordering)
            if self.use_ancilla:
                counts = [{key[1:]: value for (key, value) in count.items()} for count in counts]

            for count in counts:
                if num_qubits*'0' not in count:
                    count[num_qubits*'0'] = 0

            # The statistics give us access to the probabilities, hence to the
            # absolute overlap squared directly.
            overlap_squared = [count[num_qubits*'0'] / self._shots for count in counts]
            overlap_squared = np.array(overlap_squared)

        # If one uses a simulator, one can for instance
        # directly access the output statevectors.
        elif self._backend_name == 'statevector_simulator':

            # Preparing the state |00...0>
            computational_basis_vector = np.zeros(shape=(2**num_qubits,))
            computational_basis_vector[0] = 1
            result = self.qi.execute(circuits=circuits, had_transpiled=False)
            statevector = [np.real(result.get_statevector(i)) for i in range(len(circuits))]
            statevector = np.stack(statevector)
            overlap = np.einsum('a,ba->b', computational_basis_vector, statevector)
            overlap_squared = np.absolute(overlap)**2 # One has to square the amplitudes to get the probabilities

        overlap_squared = overlap_squared.reshape((X1.shape[0], X2.shape[0]))

        return overlap_squared


    def _image_to_circuit(
        self,
        image: np.ndarray
    ) -> QuantumCircuit:
        r"""Associates a quantum circuit to an image. The output of the circuit is the encoded image.
        Args:
            image (np.ndarray): Single image whose quantum circuit representation one is computing.
        Returns:
            (QuantumCircuit): Quatum circuit representation of image.
        """

        image_size = image.shape[0]
        side_qubits = ceil(log2(image_size))
        image_qubits = 2 * side_qubits
        self.num_qubits = image_qubits + 1

        image_register = QuantumRegister(image_qubits, 'position')
        spin_register = QuantumRegister(1, 'spin')

        if self.use_ancilla:
            ancilla_register = QuantumRegister(1, 'ancilla')
            qc = QuantumCircuit(spin_register, image_register, ancilla_register)

        else:
            qc = QuantumCircuit(spin_register, image_register)

        qc.i(0)
        for i in range(1, image_register.size + 1):
            qc.h(i)
        if self.use_ancilla:
            qc.i(image_qubits+1)

        qc.barrier()

        for i in range(image_size):
            for j in range(image_size):

                if image[i, j] == 1:

                    binarized_i = self._binary_formatting(digit=i, n_bits=side_qubits, reverse=False)
                    binarized_j = self._binary_formatting(digit=j, n_bits=side_qubits, reverse=False)

                    flip_idx = []

                    for ii in range(side_qubits):
                        if binarized_i[ii] == '1':
                            flip_idx.append(ii+1)
                            qc.x(ii+1)

                    for jj in range(side_qubits):
                        if binarized_j[jj] == '1':
                            flip_idx.append(jj+side_qubits+1)
                            qc.x(jj+side_qubits+1)

                    if self.use_ancilla:
                        qc.mcx(
                            control_qubits=list(range(1, image_qubits+1)),
                            target_qubit=0,
                            ancilla_qubits=image_qubits+1,
                            mode='recursion'
                        )

                    else:
                        qc.mcx(
                            control_qubits=list(range(1, image_qubits+1)),
                            target_qubit=0,
                            ancilla_qubits=None,
                            mode='noancilla'
                        )

                    for q in flip_idx:
                        qc.x(q)

                    qc.barrier()

        return qc

    @staticmethod
    def _binary_formatting(
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












###

            #if self.mitigate:
            #    qr = QuantumRegister(self.n_qubits)
            #    meas_calibs, state_labels = complete_meas_cal(qr=qr, circlabel='mcal')
            #    #cal_qc = transpile(meas_calibs, self.qi.backend)
            #    #qc = cal_qc + qc
            #    qc = meas_calibs + qc
#
            #    results = self.qi.execute(circuits=qc, had_transpiled=True)
#
            #    # Split the results into to Result objects for calibration and kernel computation
            #    cal_res = Result(backend_name=results.backend_name,
            #        backend_version=results.backend_version,
            #        qobj_id=results.qobj_id,
            #        job_id=results.job_id,
            #        success=results.success,
            #        #results=results.results[:len(cal_qc)]
            #        results=results.results[:len(meas_calibs)]
            #    )
#
            #    data_res = Result(backend_name=results.backend_name,
            #        backend_version=results.backend_version,
            #        qobj_id=results.qobj_id,
            #        job_id=results.job_id,
            #        success=results.success,
            #        #results=results.results[len(cal_qc):]
            #        results=results.results[len(meas_calibs):]
            #    )
#
            #    # Apply measurement calibration and computer the calibration filter
            #    meas_filter = CompleteMeasFitter(cal_res, state_labels, circlabel='mcal').filter
#
            #    # Apply the calibration filter to the results and get the counts
            #    mitigated_results = meas_filter.apply(data_res)
            #    counts = mitigated_results.get_counts()
#
            #else: