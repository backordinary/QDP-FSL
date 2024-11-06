# https://github.com/QSciTech-QuantumBC-Workshop/Group5/blob/4fdf4d91611e642c638cf3d89daca150992b0bf1/estimator.py
"""
estimator.py - To estimate expectation value of observables

Copyright 2020-2023 Maxime Dion <maxime.dion@usherbrooke.ca>
This file has been modified by <Your,Name> during the
QSciTech-QuantumBC virtual workshop on gate-based quantum computing.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import time

import numpy as np

from qiskit import QuantumCircuit, execute, ClassicalRegister
from qiskit.providers import Backend
from qiskit.circuit import Instruction

from pauli_string import PauliString, LinearCombinaisonPauliString

from typing import Union, Optional
from numpy.typing import NDArray


class Estimator:
    def __init__(self, varform: QuantumCircuit, backend: Backend, execute_opts={}, record: Optional[object] = None,
                mitigate: Optional[object] = False):
        """
        An Estimator allows to transform an observable into a callable function. The observable is not set at the 
        initialization. The estimator will build the QuantumCircuit necessary to estimate the expected value of the 
        observable. Upon using the 'eval' method, it will execute these circuits and interpret the results to return an
        estimate of the expected value of the observable.

        Args:
            varform (QuantumCircuit): A paramatrized QuantumCircuit.
            backend (Backend): A qiskit backend. Could be a simulator are an actual quantum computer.
            execute_opts (dict, optional): Optional arguments to be passed to the qiskit.execute function.
                                           Defaults to {}.

            record (object, optional): And object that could be called on each evaluation to record the results. 
                Defaults to None.
        """

        self.varform = varform
        self.backend = backend
        self.execute_opts = execute_opts

        self.record = record
        self.mitigate = mitigate

        # To be set attributes
        self.n_qubits = varform.num_qubits
        self.diagonalizing_circuits = list()
        self.diagonal_observables = list()

    def set_observable(self, observable: LinearCombinaisonPauliString) -> None:
        """
        Set the observable which the expectation value will be estimated.
        This sets the value of the attribute 'n_qubits'.
        The observable is converted into a list of diagonal observables along the circuit which performs this diagonalization.
        This is done using the 'diagonal_observables_and_circuits' method (defined at the subclass level).

        Args:
            observable (LinearCombinaisonPauliString): The observable to be evaluated.
        """

        self.diagonal_observables, self.diagonalizing_circuits = self.diagonal_observables_and_circuits(observable)

    def eval(self, params: Union[NDArray, list]) -> float:
        """
        Evaluate an estimation of the expectation value of the set observable.

        Args:
            params (list or NDArray): Parameter values at which the expectation value should be evaluated.
                Will be fed to the 'varform' paramatrized QuantumCircuit.

        Returns:
            float: The estimated expectation value of the observable.
        """

        t0 = time.time()
        state_circuit = self.prepare_state_circuit(params)
        circuits = self.assemble_circuits(state_circuit)
        
        expectation_value = 0
        ################################################################################################################
        # YOUR CODE HERE
        # TO COMPLETE (after lecture on VQE)
        # A. For each pair of diagonal observable and circuit :
        #   1. Execute the circuit on the backend
        #   2. Extract the result from the job
        #   3. Estimate the expectation value of the diagonal_observable
        # B. Combine all the results into the expectation value of the observable (e.i. the energy)
        # (Optional) record the result with the record object
        # (Optional) monitor the time of execution        
        for i in range(len(circuits)):
            job = execute(circuits[i], backend=self.backend, **self.execute_opts)
            # print(self.execute_opts)
            result = job.result()
            counts = result.get_counts(circuits[i])
            if self.mitigate:
                err_matrix = np.asarray(self.execute_opts['noise_model'].to_dict()['errors'][-1]['probabilities'])
                # print((err_matrix @ np.linalg.inv(err_matrix)).sum().round(), end='')            
                vec_count = np.zeros(2 ** self.n_qubits)
                for s in range(2 ** self.n_qubits):
                    if f'{s:04b}' in counts:
                        vec_count[s] = counts[f'{s:04b}']                        
                miti_vec_count = np.linalg.inv(err_matrix) @ vec_count
                
                for s in range(2 ** self.n_qubits):
                    counts[f'{s:04b}'] = miti_vec_count[s]
                
            # else:
            #     print(0, end='')
            v = self.estimate_diagonal_pauli_string_expectation_value(self.diagonal_observables[i].pauli_strings[0], counts)
            expectation_value += v * self.diagonal_observables[i].coefs[0]                  ################################################################################################################

        # raise NotImplementedError()

        eval_time = time.time()-t0
        # print(f'evaluation time = {eval_time} seconds')
        self.record = expectation_value

        return expectation_value
    
    
    def prepare_state_circuit(self, params: Union[NDArray, list]) -> QuantumCircuit:
        """
        Assign parameter values to the variational circuit (varfom) to prepare the quantum state.

        Args:
            params (list or NDArray): Params to be assigned to the 'varform' QuantumCircuit.

        Returns:
            QuantumCircuit: The quantum state circuit
        """

        state_circuit = QuantumCircuit(self.n_qubits)

        ################################################################################################################
        # YOUR CODE HERE
        # TO COMPLETE (after lecture on VQE)
        param_dict = dict(zip(self.varform.parameters, params))
        state_circuit = self.varform.assign_parameters(param_dict)
        ################################################################################################################

        # raise NotImplementedError()

        return state_circuit

    def assemble_circuits(self, state_circuit: QuantumCircuit) -> list[QuantumCircuit]:
        """
        For every diagonal observable, assemble the complete circuit with:
        - State preparation
        - Measurement circuit
        - Measurements

        Args:
            state_circuit (QuantumCircuit): The quantum state circuit

        Returns:
            list<QuantumCircuit>: The quantum circuits to be executed.
        """

        circuits = list()
        ################################################################################################################
        # YOUR CODE HERE
        # TO COMPLETE (after lecture on VQE)
        state_circuit.barrier(range(self.n_qubits))
        for i in range(len(self.diagonal_observables)):
            cr = ClassicalRegister(4)
            qc = state_circuit.compose(self.diagonalizing_circuits[i])
            # qc.measure_all()
            qc.add_register(cr)
            # qc.measure(range(4),range(4))
            qc.append(Instruction("measure", 4, 4, []), range(4), range(4))
            # meas = Instruction("measure", 4, 4, [])
            # # print(meas)
            # qc.append(meas, [0, 1, 2, 3], [0, 1, 2, 3])
            circuits.append(qc)           
            
        ################################################################################################################

        # raise NotImplementedError()

        return circuits

    @staticmethod
    def diagonal_pauli_string_eigenvalue(diagonal_pauli_string: PauliString, state: str) -> float:
        """
        Computes the eigenvalue (+1 or -1) of a diagonal pauli string for a basis state.

        Args:
            diagonal_pauli_string (PauliString): A diagonal pauli string
            state (str): a basis state (ex : '1011')

        Returns:
            float: The eigenvalue
        """

        eigenvalue = 0

        ################################################################################################################
        # YOUR CODE HERE
        # TO COMPLETE (after lecture on VQE)
        for i in range(len(state)):
            if str(diagonal_pauli_string)[i] == 'Z' and state[i] == '1':
                eigenvalue += 1
                # print(eigenvalue)
        eigenvalue = (-1) ** (eigenvalue % 2)
        ################################################################################################################

        # raise NotImplementedError()

        return eigenvalue

    @staticmethod
    def estimate_diagonal_pauli_string_expectation_value(diagonal_pauli_string: PauliString, counts: dict) -> float:
        """
        Estimate the expectation value for a diagonal pauli string based on counts.

        Args:
            diagonal_pauli_string (PauliString): The diagonal pauli string (must be only I and Z)
            counts (dict): Contains the number of times each basis state was obtained from a measurement

        Returns:
            float: The expectation value of the Pauli string
        """

        expectation_value = 0

        ################################################################################################################
        # YOUR CODE HERE
        # TO COMPLETE (after lecture on VQE)
        total = sum(counts.values())
        for state in counts.keys():
            l = Estimator.diagonal_pauli_string_eigenvalue(diagonal_pauli_string, state)
            expectation_value += l * counts[state]
        expectation_value = expectation_value / total
        ################################################################################################################

        # raise NotImplementedError()

        return expectation_value

    @staticmethod
    def estimate_diagonal_observable_expectation_value(diagonal_observable: LinearCombinaisonPauliString,
                                                       counts: dict) -> float:
        """
        Estimate the expectation value for a diagonal observable (linear combinaison of pauli strings) based on counts.

        Args:
            diagonal_observable (LinearCombinaisonPauliString): The observable (must be only I and Z)
            counts (dict): Contains the number of times each basis state was obtained from a measurement

        Returns:
            float: The expectation value of the Observable
        """

        expectation_value = 0

        ################################################################################################################
        # YOUR CODE HERE
        # TO COMPLETE (after lecture on VQE)
        for i in range(len(diagonal_observable)):
            v = Estimator.estimate_diagonal_pauli_string_expectation_value(diagonal_observable.pauli_strings[i], counts)
            expectation_value += diagonal_observable.coefs[i] * v
            # print(v)
        ################################################################################################################

        # raise NotImplementedError()

        return expectation_value

    @staticmethod
    def diagonalizing_pauli_string_circuit(pauli_string: PauliString) -> tuple[QuantumCircuit, PauliString]:
        """
        Builds the circuit representing the transformation which diagonalizes the given PauliString.

        Args:
            pauli_string (PauliString): The pauli string to be diagonalized

        Returns:
            QuantumCircuit: A quantum circuit representing the transformation which diagonalizes the given PauliString
            PauliString: The diagonal PauliString                   
        """
        
        n_qubits = len(pauli_string)
        diagonalizing_circuit = QuantumCircuit(n_qubits)
        diagonal_pauli_string = ''

        ################################################################################################################
        # YOUR CODE HERE
        # TO COMPLETE (after lecture on VQE)
        for i in range(n_qubits):
            bit = str(pauli_string)[i]
            if bit == 'X':
                diagonalizing_circuit.h(n_qubits - i - 1)
                diagonal_pauli_string += 'Z'
            elif bit == 'Y':
                diagonalizing_circuit.sdg(n_qubits - i - 1)
                diagonalizing_circuit.h(n_qubits - i - 1)
                diagonal_pauli_string += 'Z'
            else:
                diagonal_pauli_string += bit
            # print(diagonal_pauli_string)

        PauliString.from_str(diagonal_pauli_string)
        ################################################################################################################
        
        # raise NotImplementedError()

        return diagonalizing_circuit, diagonal_pauli_string


class BasicEstimator(Estimator):
    """
    The BasicEstimator should build 1 quantum circuit and 1 interpreter for each PauliString.
    The interpreter should be 1d array of size 2**number of qubits.
    It does not exploit the fact that commuting PauliStrings can be evaluated from a common circuit.
    """
    
    @staticmethod
    def diagonal_observables_and_circuits(observable: LinearCombinaisonPauliString) -> tuple[list[LinearCombinaisonPauliString],
                                                                                             list[QuantumCircuit]]:
        """
        This method converts each PauliString in the observable into :
        - A diagonal observable including the associated coefficient
        - The quantum circuit that performs this diagonalization.
        The diagonal observables and the quantum circuits are return as two list.

        Args:
            observable (LinearCombinaisonPauliString): An observable.

        Returns:
            list<LinearCombinaisonPauliString> : The diagonal observables (one PauliString long).
            list<list of QuantumCircuit> : The diagonalizing quantum circuits
        """
        
        diagonal_observables = list()
        diagonalizing_circuits = list()
        
        ################################################################################################################
        # YOUR CODE HERE
        # TO COMPLETE (after lecture on VQE)
        # Hint : the next method does the work for 1 PauliString + coef
        strings = observable.pauli_strings
        coefs = observable.coefs
        for i in range(len(observable)):
            (diagonalizing_circuit, diagonal_pauli_string) = Estimator.diagonalizing_pauli_string_circuit(strings[i])
            diagonal_observables.append(LinearCombinaisonPauliString([coefs[i]], [diagonal_pauli_string]))
            diagonalizing_circuits.append(diagonalizing_circuit)
        ################################################################################################################
        
        # raise NotImplementedError()

        return diagonal_observables, diagonalizing_circuits


class BitwiseCommutingCliqueEstimator(Estimator):
    """
    The BitwiseCommutingCliqueEstimator should build 1 quantum circuit and 1 interpreter for each clique of PauliStrings.
    The interpreter should be 2d array of size (number of cliques ,2**number of qubits).
    It does exploit the fact that commuting PauliStrings can be evaluated from a common circuit.
    """

    @staticmethod
    def diagonal_observables_and_circuits(observable: LinearCombinaisonPauliString) -> tuple[list[LinearCombinaisonPauliString],
                                                                                             list[QuantumCircuit]]:
        """
        This method first divide the observable into bitwise commuting cliques. Each commuting clique is then converted
        into :
        - A diagonal observable including the associated coefficients
        - The quantum circuit that performs this diagonalization.
        The diagonal observables and the quantum circuits are return as two list.

        Args:
            observable (LinearCombinaisonPauliString): An observable.

        Returns:
            list<LinearCombinaisonPauliString> : The diagonal observables
            list<list of QuantumCircuit> : The diagonalizing quantum circuits
        """

        cliques = observable.divide_in_bitwise_commuting_cliques()

        diagonal_observables = list()
        diagonalizing_circuits = list()
        
        ################################################################################################################
        # YOUR CODE HERE
        # TO COMPLETE (after lecture on VQE)
        # Hint : the next method does the work for 1 PauliString + coef
        ################################################################################################################
        
        raise NotImplementedError()
            
        return diagonal_observables, diagonalizing_circuits


