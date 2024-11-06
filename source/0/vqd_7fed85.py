# https://github.com/nahumsa/volta/blob/77c060b56162ca7eb0072a594f5470d75ccb81ed/volta/vqd.py
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


import numpy as np

from typing import Union


from qiskit import QuantumCircuit
from qiskit.opflow import OperatorBase, ListOp, PrimitiveOp, PauliOp
from qiskit.algorithms.optimizers import Optimizer
from qiskit.aqua import QuantumInstance
from qiskit.providers import BaseBackend


from volta.observables import sample_hamiltonian
from volta.swaptest import (
    measure_swap_test,
    measure_dswap_test,
    measure_amplitude_transition_test,
)


class VQD(object):
    """Variational Quantum Deflation algorithm class.

    Based on https://arxiv.org/abs/1805.08138

    """

    def __init__(
        self,
        hamiltonian: Union[OperatorBase, ListOp, PrimitiveOp, PauliOp],
        ansatz: QuantumCircuit,
        n_excited_states: int,
        beta: float,
        optimizer: Optimizer,
        backend: Union[BaseBackend, QuantumInstance],
        overlap_method: str = "swap",
        num_shots: int = 10000,
        debug: bool = False,
    ) -> None:
        """Initialize the class.

        Args:
            hamiltonian (Union[OperatorBase, ListOp, PrimitiveOp, PauliOp]): Hamiltonian
            constructed using qiskit's aqua operators.
            ansatz (QuantumCircuit): Anstaz that you want to run VQD.
            n_excited_states (int): Number of excited states that you want to find the energy
            if you use 0, then it is the same as using a VQE.
            beta (float): Strenght parameter for the swap test.
            optimizer (qiskit.algorithms.optimizers.Optimizer): Classical Optimizers
            from Terra.
            backend (Union[BaseBackend, QuantumInstance]): Backend for running the algorithm.
            overlap_method (str): State overlap method. Methods available: swap, dswap, amplitude (Default: swap)
            num_shots (int): Number of shots. (Default: 10000)
        """

        # Input parameters
        self.hamiltonian = hamiltonian
        self.n_qubits = hamiltonian.num_qubits
        self.optimizer = optimizer
        self.backend = backend
        self.NUM_SHOTS = num_shots
        self.BETA = beta
        self.overlap_method = overlap_method

        IMPLEMENTED_OVERLAP_METHODS = ["swap", "dswap", "amplitude"]
        if self.overlap_method not in IMPLEMENTED_OVERLAP_METHODS:
            raise NotImplementedError(
                f"overlapping method not implemented. Available implementing methods: {IMPLEMENTED_OVERLAP_METHODS}"
            )

        # Helper Parameters
        self.n_excited_states = n_excited_states + 1
        self.ansatz = ansatz
        self.n_parameters = self._get_num_parameters
        self._debug = debug

        # Logs
        self._states = []
        self._energies = []

    @property
    def energies(self) -> list:
        """Returns a list with energies.

        Returns:
            list: list with energies
        """
        return self._energies

    @property
    def states(self) -> list:
        """Returns a list with states associated with each energy.

        Returns:
            list: list with states.
        """
        return self._states

    @property
    def _get_num_parameters(self) -> int:
        """Get the number of parameters in a given ansatz.

        Returns:
            int: Number of parameters of the given ansatz.
        """
        return len(self.ansatz.parameters)

    def _apply_varform_params(self, params: list):
        """Get an hardware-efficient ansatz for n_qubits
        given parameters.
        """

        # Define variational Form
        var_form = self.ansatz

        # Get Parameters from the variational form
        var_form_params = sorted(var_form.parameters, key=lambda p: p.name)

        # Check if the number of parameters is compatible
        assert len(var_form_params) == len(
            params
        ), "The number of parameters don't match"

        # Create a dictionary with the parameters and values
        param_dict = dict(zip(var_form_params, params))

        # Assing those values for the ansatz
        wave_function = var_form.assign_parameters(param_dict)

        return wave_function

    def cost_function(self, params: list) -> float:
        """Evaluate the cost function of VQD.

        Args:
            params (list): Parameter values for the ansatz.

        Returns:
            float: Cost function value.
        """
        # Define Ansatz
        qc = self._apply_varform_params(params)

        # Hamiltonian
        hamiltonian_eval = sample_hamiltonian(
            hamiltonian=self.hamiltonian, ansatz=qc, backend=self.backend
        )

        # Fidelity
        fidelity = 0.0
        if len(self.states) != 0:
            for state in self.states:
                if self.overlap_method == "dswap":
                    swap = measure_dswap_test(qc, state, self.backend, self.NUM_SHOTS)
                elif self.overlap_method == "swap":
                    swap = measure_swap_test(qc, state, self.backend, self.NUM_SHOTS)
                elif self.overlap_method == "amplitude":
                    swap = measure_amplitude_transition_test(
                        qc, state, self.backend, self.NUM_SHOTS
                    )
                fidelity += swap

                if self._debug:
                    print(fidelity)

        # Get the cost function
        cost = hamiltonian_eval + self.BETA * fidelity

        return cost

    def optimizer_run(self):

        # Random initialization
        params = np.random.rand(self.n_parameters)

        optimal_params, energy, n_iters = self.optimizer.optimize(
            num_vars=self.n_parameters,
            objective_function=self.cost_function,
            initial_point=params,
        )

        # Logging the energies and states
        # TODO: Change to an ordered list.

        self._energies.append(energy)
        self._states.append(self._apply_varform_params(optimal_params))

    def _reset(self):
        """Resets the energies and states helper variables."""

        self._energies = []
        self._states = []

    def run(self, verbose: int = 1):

        self._reset()

        for i in range(self.n_excited_states):

            if verbose == 1:
                print(f"Calculating excited state {i}")

            self.optimizer_run()
