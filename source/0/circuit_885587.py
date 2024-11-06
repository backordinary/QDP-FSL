# https://github.com/iamtxena/quantum-channel-discrimination/blob/ef0ade99922aa57a4c3a91876b1a2e56d079c67a/qcd/circuits/circuit.py
from abc import ABC, abstractmethod

from qiskit.circuit.classicalregister import ClassicalRegister
from qcd.backends.statevector_simulator import StateVectorSimulatorBackend
from qiskit import QuantumCircuit, execute
from qiskit.result.result import Result
from qcd.configurations.configuration import ChannelConfiguration
from qcd.backends import DeviceBackend, SimulatorBackend
from typing import Optional, Tuple, cast, List, Dict
from ..typings.configurations import Fidelities, OptimalConfigurations, ValidatedConfiguration
from .aux import set_only_eta_groups, fix_configurations
import numpy as np
import time


class Circuit(ABC):
    """ Generic class acting as an interface for any Quantum Damping Channel Circuit"""

    def __init__(self,
                 optimal_configurations: Optional[OptimalConfigurations] = None,
                 backend: Optional[DeviceBackend] = SimulatorBackend()):
        self._backend = SimulatorBackend() if backend is None else backend
        self._statevector_backend = StateVectorSimulatorBackend()

        if optimal_configurations is not None:
            """ !!! THIS IS ANOTHER FIX: to be able to read legacy format configuration """
            self._optimal_configurations = set_only_eta_groups(cast(List[Dict], [optimal_configurations])).pop()
            """ !!! THIS IS A FIX: to be able to read legacy format configuration """
            self._optimal_configurations = fix_configurations(optimal_configurations)

    def one_shot_run(self, plays: Optional[int] = 100) -> OptimalConfigurations:
        """ Runs all the experiments using the optimal configurations and computing the success probability """
        if self._optimal_configurations is None:
            raise ValueError('Optimal Configurations must be provided on Circuit Constructor to run experiments')

        total_configurations = len(self._optimal_configurations['configurations'])
        optimal_results = self.init_optimal_results(total_configurations)
        program_start_time = time.time()

        print(f"Starting the computation for {total_configurations} configurations.")
        for idx, configuration in enumerate(self._optimal_configurations['configurations']):
            optimal_results['probabilities'][idx] = self.compute_average_success_probability(
                configuration=configuration,
                plays=plays)

            self.print_computation_time(total_configurations, optimal_results, program_start_time, idx, configuration)

        end_time = time.time()
        print("total minutes of execution time: " +
              f'{np.round((end_time - program_start_time)/60, 0)} minutes' +
              f' and {np.round((end_time - program_start_time) % 60, 0)} seconds')
        self._optimal_results = optimal_results
        return optimal_results

    def init_optimal_results(self, total_configurations):
        optimal_results: OptimalConfigurations = {
            'eta_groups': self._optimal_configurations['eta_groups'],
            'best_algorithm': self._optimal_configurations['best_algorithm'],
            'probabilities': [0] * total_configurations,
            'configurations': self._optimal_configurations['configurations'],
            'number_calls_made': [1] * total_configurations,
            'legacy': (True if
                       ('legacy' in self._optimal_configurations and
                        self._optimal_configurations['legacy'] is True)
                       else False)}
        return optimal_results

    def print_computation_time(self,
                               total_configurations,
                               optimal_results,
                               program_start_time,
                               idx,
                               configuration):
        end_time = time.time()
        if idx % 30 == 0 and (end_time - program_start_time <= 60):
            print(f"Configuration # {idx} of {total_configurations}, time from start: " +
                  f'{np.round((end_time - program_start_time), 0)} seconds')
        if idx % 30 == 0 and (end_time - program_start_time > 60):
            print(f"Configuration # {idx} of {total_configurations}, time from start: " +
                  f'{np.round((end_time - program_start_time)/60, 0)} minutes' +
                  f' and {np.round((end_time - program_start_time) % 60, 0)} seconds')
        if idx % 30 == 0:
            print('computed ',
                  (self._create_one_configuration(configuration, sorted(configuration.eta_group)).to_dict()))
            print(f"Configuration index: {idx}, Probabilities ->  computed: " +
                  f"{optimal_results['probabilities'][idx]}, " +
                  f"optimized: {self._optimal_configurations['probabilities'][idx]} and " +
                  "Delta: ",
                  np.round(optimal_results['probabilities'][idx] -
                           self._optimal_configurations['probabilities'][idx], 2))

    def compute_average_success_probability(self,
                                            configuration: ChannelConfiguration,
                                            plays: Optional[int] = 100) -> float:
        """ Computes the average success probability of running a specific configuration
            for the number of plays defined in the configuration and for each eta from
            the eta group
        """
        if plays is None:
            plays = 100

        self._backend = SimulatorBackend() if self._backend is None else self._backend
        eta_counts = self._get_eta_counts(configuration, plays)

        return self._guess_probability_from_counts(eta_counts, plays, len(configuration.eta_group))

    def _get_eta_counts(self, configuration, plays):
        eta_counts = [cast(Result, execute(self._create_one_circuit(configuration, eta),
                                           backend=self._backend.backend,
                                           shots=plays).result()).get_counts()
                      for eta in configuration.eta_group]
        return eta_counts

    def _create_one_circuit(self,
                            configuration: ChannelConfiguration,
                            eta: float) -> QuantumCircuit:
        creg_c, circuit = self._create_one_circuit_without_measurement(configuration, eta)
        return self._add_measurement_to_one_circuit(creg_c, circuit)

    def validate_optimal_configuration(self,
                                       configuration: ChannelConfiguration,
                                       plays: Optional[int] = 10000) -> ValidatedConfiguration:
        """ Runs the circuit with the given optimal configuration computing the success average probability
            for each eta (and also the global), the selected eta for each measured state and finally the
            upper and lower bound fidelities
        """
        if plays is None:
            plays = 10000

        circuits = [self._create_one_circuit_without_measurement(configuration=configuration, eta=eta)
                    for eta in configuration.eta_group]
        fidelities = self._run_circuit_for_each_eta_get_fidelities(circuits=circuits)
        validated_configuration = self._run_circuit_for_each_eta_get_probabilities_and_etas_assigned(
            configuration=configuration,
            circuits=circuits,
            plays=plays)
        validated_configuration['fidelities'] = fidelities

        return validated_configuration

    def _run_circuit_for_each_eta_get_probabilities_and_etas_assigned(
            self,
            configuration: ChannelConfiguration,
            circuits: List[Tuple[ClassicalRegister, QuantumCircuit]],
            plays: int) -> ValidatedConfiguration:
        """ Runs the circuit with the given optimal configuration computing the success average probability
            for each eta (and also the global) and the selected eta for each measured state
        """
        eta_counts = [cast(Result, execute(self._add_measurement_to_one_circuit(creg_c, circuit),
                                           backend=self._backend.backend,
                                           shots=plays).result()).get_counts()
                      for creg_c, circuit in circuits]
        return self._get_probabilities_and_etas_assigned_from_counts(eta_counts, plays, len(configuration.eta_group))

    def _run_circuit_for_each_eta_get_fidelities(
            self,
            circuits: List[Tuple[ClassicalRegister, QuantumCircuit]]) -> Fidelities:
        """ Runs the circuit with the given optimal configuration using a state vector backend
            and computes the fidelity for each pair of channels
        """
        state_vectors = [cast(Result, execute(circuit,
                                              backend=self._statevector_backend.backend).result()).get_statevector()
                         for _, circuit in circuits]
        return self._compute_upper_and_lower_fidelity_bounds(state_vectors=state_vectors)

    @abstractmethod
    def _guess_probability_from_counts(self, counts: List[dict], plays: int,
                                       eta_group_length: int) -> float:
        """ Computes the probability from the 'counts' measured
            based on the guess strategy that is required to use
        """
        pass

    @abstractmethod
    def _get_probabilities_and_etas_assigned_from_counts(self, counts: List[dict], plays: int,
                                                         eta_group_length: int) -> ValidatedConfiguration:
        """ Computes the validated probability from the 'counts' measured
            based on the guess strategy that is required to use and returns the
            etas assigned for each measured state
        """
        pass

    @abstractmethod
    def _create_one_configuration(self, configuration: ChannelConfiguration,
                                  eta_group: List[float]) -> ChannelConfiguration:
        """ Creates a specific configuration setting a specific eta pair """
        pass

    @abstractmethod
    def _create_one_circuit_without_measurement(self,
                                                configuration: ChannelConfiguration,
                                                eta: float) -> Tuple[ClassicalRegister, QuantumCircuit]:
        """ Creates one circuit from a given configuration and eta without measurement gates """
        pass

    @abstractmethod
    def _compute_upper_and_lower_fidelity_bounds(self, state_vectors: List[np.ndarray]) -> Fidelities:
        """ Computes upper and lower fidelity bounds from the given state vectors """
        pass

    @abstractmethod
    def _add_measurement_to_one_circuit(self,
                                        creg_c: ClassicalRegister,
                                        circuit: QuantumCircuit) -> QuantumCircuit:
        pass
