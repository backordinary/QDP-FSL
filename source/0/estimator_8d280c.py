# https://github.com/QSciTech-QuantumBC-Workshop/Group3/blob/bc3ba273d16bc79079a18637f285173c4712f355/molecule-incomplete-suggested-solution-main/estimator.py
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

from qiskit import QuantumCircuit, execute
from qiskit.providers import Backend

from pauli_string import PauliString, LinearCombinaisonPauliString

from typing import Union, Optional, List, Tuple
from numpy.typing import NDArray


class Estimator:
    def __init__(self, varform: QuantumCircuit, backend: Backend, execute_opts={}, record: Optional[object] = None):
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

    def eval(self, params: Union[NDArray, List]) -> float:
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
        
        diagonal_observables = self.diagonal_observables
        job = execute(circuits, backend = self.backend, **self.execute_opts)
        result = job.result()
        
#         i = 0
#         counts = result.get_counts(circuits[i])
#         for i, diag_observ in enumerate(diagonal_observables):
#             expectation_value += Estimator.estimate_diagonal_observable_expectation_value(diag_observ, counts)
            
        counts_list = list()    
        for i, diag_observ in enumerate(diagonal_observables):
            counts = result.get_counts(circuits[i])
            counts_list.append(counts)
#             expectation_value += Estimator.estimate_diagonal_pauli_string_expectation_value(diag_observ, counts)
            expectation_value += Estimator.estimate_diagonal_observable_expectation_value(diag_observ, counts)

        ################################################################################################################

#         raise NotImplementedError()
#         print(diagonal_observables)
#         print(counts_list)
        self.counts_list = counts_list
        self.dig_ob = diagonal_observables
        self.cir = circuits

        eval_time = time.time()-t0

        return expectation_value

    def prepare_state_circuit(self, params: Union[NDArray, List]) -> QuantumCircuit:
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
        varform_qc = self.varform
        param_dict = dict(zip(varform_qc.parameters, params))
        state_circuit = varform_qc.assign_parameters(param_dict)
        
        ################################################################################################################

#         raise NotImplementedError()

        return state_circuit

    def assemble_circuits(self, state_circuit: QuantumCircuit) -> List[QuantumCircuit]:
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
       
        diag_qcircuit_list  = self.diagonalizing_circuits

        #assembles the state circuit to all the diagonalized oberervable circuits (non-unique circuits in circuit list)
#         for i in range(len(diag_qcircuit_list)):
#             diag_qcircuit_list[i].measure_all()
#             state_circuit.compose(diag_qcircuit_list[i], inplace = True)
#             circuits.append(state_circuit)
            
        #assembles the state circuit to each diagonazlized observable circuits (unique circuits in circuit list)
#         for i in range(len(diag_qcircuit_list)):
#             state_circuit.barrier()
#             c = state_circuit + diag_qcircuit_list[i]
#             c.measure_all()
#             circuits.append(c)
            
            
        for i in range(len(diag_qcircuit_list)):
#             state_circuit.barrier()
            c = state_circuit + diag_qcircuit_list[i]
#             c.measure_all()
            circuits.append(c)
        
        measured_circuits = list()
        for i in range(len(circuits)):
            circuits[i].measure_all()
            measured_circuits.append(circuits[i])
            
            
        ################################################################################################################

#         raise NotImplementedError()

#         return circuits
        return measured_circuits#circuits

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

        eigenvalue = 0.


        ################################################################################################################
        # YOUR CODE HERE
        # TO COMPLETE (after lecture on VQE)
        
        #### See page 19 of VQE.pdf file.  <q | diagonal_pauli_string | q> = (-1)^(number of matching 1 and Z)
        
        basis_state = state
        diag_pstring = diagonal_pauli_string.__str__()
        
        no_1_match_Z = 0.
        for i, st in enumerate(diag_pstring):
            if st == 'Z' and basis_state[i] == '1':
                no_1_match_Z += 1.
            else:
                no_1_match_Z += 0.
                
        eigenvalue += (-1)**no_1_match_Z 
                
                
        ################################################################################################################

#         raise NotImplementedError()

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

        expectation_value = 0.

        ################################################################################################################
        # YOUR CODE HERE
        # TO COMPLETE (after lecture on VQE)
        
        total_counts = 0.
        for basis_state, count_value in counts.items():
            eig_val = Estimator.diagonal_pauli_string_eigenvalue(diagonal_pauli_string, basis_state)
            expectation_value += (count_value*eig_val)
            total_counts += count_value
            
        expectation_value = expectation_value/total_counts
        
        ################################################################################################################

#         raise NotImplementedError()

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

        expectation_value = 0.

        ################################################################################################################
        # YOUR CODE HERE
        # TO COMPLETE (after lecture on VQE)
        
        coeff_list = diagonal_observable.coefs
        coeff_list = (np.abs(coeff_list))
        
        
#         t = diagonal_observable.coefs
#         if all(t.imag == 0.):           # if coefficients are all real, take absolute value (=|x|)
#             coeff_list = t.real
            
#         else:  
#             if all(t.imag != 0.):      # if coefficients are all imaginary, take the square of the absoulte value (=x^2 + y^2)
#                 coeff_list = (np.abs(t))**2

#             if any(t.imag != 0.):    # if some are real and some others are imaginary
#                 coeff_list = list()
#                 for i, cf in enumerate(t):
#                     if cf.imag == 0.:
#                         coeff_list.append(t.real)   #if real take the real part
#                     else:
#                         coeff_list.append((np.abs(cf))**2) #take the square of the absoulte value (=x^2 + y^2)

        coeff_list = diagonal_observable.coefs.real
        diag_pauli_string_list = diagonal_observable.pauli_strings
        
        for i, p_string in enumerate(diag_pauli_string_list):
            expectation_value += coeff_list[i]*Estimator.estimate_diagonal_pauli_string_expectation_value(p_string, counts)
#             print(expectation_value)
            
        ################################################################################################################

#         raise NotImplementedError()
#         print(diag_pauli_string_list[0])
        return expectation_value

    @staticmethod
    def diagonalizing_pauli_string_circuit(pauli_string: PauliString) -> Tuple[QuantumCircuit, PauliString]:
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
        diagonal_pauli_string = None

        ################################################################################################################
        # YOUR CODE HERE
        # TO COMPLETE (after lecture on VQE)
        
        #### Implementing slide 23 of VQE.pdf 
        
        string = pauli_string.__str__()  #convert PauliString object to actual string
        string = string[::-1]       # reverse string
        
        new_string = ''
        
        for  i, st in enumerate(string):
            if st == 'X':
                new_string += 'Z'
                diagonalizing_circuit.h(i)
            if st == 'Y':
                new_string += 'Z'
                diagonalizing_circuit.sdg(i)
                diagonalizing_circuit.h(i)
            if st == 'Z':
                new_string += st
            if st == 'I':
                new_string += st
                
        diagonal_pauli_string = PauliString.from_str(new_string[::-1])
        
        
        
        
        
        ################################################################################################################
        
#         raise NotImplementedError()

        return diagonalizing_circuit, diagonal_pauli_string


class BasicEstimator(Estimator):
    """
    The BasicEstimator should build 1 quantum circuit and 1 interpreter for each PauliString.
    The interpreter should be 1d array of size 2**number of qubits.
    It does not exploit the fact that commuting PauliStrings can be evaluated from a common circuit.
    """
    
    @staticmethod
    def diagonal_observables_and_circuits(observable: LinearCombinaisonPauliString) -> Tuple[List[LinearCombinaisonPauliString],
                                                                                             List[QuantumCircuit]]:
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
        pl_str_list = observable.pauli_strings
        coeff_list  = observable.coefs
        
        
        for i, ps in enumerate(pl_str_list):
            diag_qcircuit, diag_ps = Estimator.diagonalizing_pauli_string_circuit(ps)
            diagonalizing_circuits.append(diag_qcircuit)
            diagonal_observables.append(coeff_list[i]*diag_ps)
        
        ################################################################################################################
        
#         raise NotImplementedError()

        return diagonal_observables, diagonalizing_circuits


class BitwiseCommutingCliqueEstimator(Estimator):
    """
    The BitwiseCommutingCliqueEstimator should build 1 quantum circuit and 1 interpreter for each clique of PauliStrings.
    The interpreter should be 2d array of size (number of cliques ,2**number of qubits).
    It does exploit the fact that commuting PauliStrings can be evaluated from a common circuit.
    """

    @staticmethod
    def diagonal_observables_and_circuits(observable: LinearCombinaisonPauliString) -> Tuple[List[LinearCombinaisonPauliString],
                                                                                             List[QuantumCircuit]]:
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
        
#         raise NotImplementedError()
            
        return diagonal_observables, diagonalizing_circuits


