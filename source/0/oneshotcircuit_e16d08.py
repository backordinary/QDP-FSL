# https://github.com/iamtxena/quantum-channel-discrimination/blob/a384db43a689fa382f49298626327d9fe03d2f44/qcd/circuits/oneshotcircuit.py

from qcd.typings.configurations import (Fidelities, MeasuredStatesCountsOneQubit,
                                        MeasuredStatesEtaAssignmentOneQubit, ValidatedConfiguration)
from qcd.configurations import OneShotConfiguration
from qcd.configurations.configuration import ChannelConfiguration
from . import Circuit
from typing import List, Tuple, cast
import numpy as np
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
import itertools
from qutip import fidelity, Qobj
from qiskit.quantum_info.states.utils import partial_trace
from .aux import upper_bound_fidelity, lower_bound_fidelity
import random


class OneShotCircuit(Circuit):
    """ Representation of the One Shot Channel circuit """

    def _prepare_initial_state(self, state_probability: float) -> Tuple[complex, complex]:
        """ Prepare initial state: computing 'x' as the amplitudes """
        return (np.sqrt(1 - state_probability), np.sqrt(state_probability))

    def _get_max_counts_distribution_for_all_channels(
            self,
            all_channel_counts: List[dict],
            counts_distribution: List[float]) -> List[float]:
        """ returns the max counts between the max counts up to that moment and the circuit counts """
        if not all_channel_counts:
            return counts_distribution

        one_channel_counts = all_channel_counts.pop()
        max_counts = [0.0, 0.0]
        if '0' in one_channel_counts:
            max_counts[0] = max([counts_distribution[0], one_channel_counts['0']])
        if '1' in one_channel_counts:
            max_counts[1] = max([counts_distribution[1], one_channel_counts['1']])

        return self._get_max_counts_distribution_for_all_channels(all_channel_counts, max_counts)

    def _guess_probability_from_counts(self,
                                       eta_counts: List[dict],
                                       plays: int,
                                       eta_group_length: int) -> float:
        """ Decides which eta was used on the real execution from the 'counts' measured
            based on the guess strategy that is required to use
        """
        counts_distribution = self._get_max_counts_distribution_for_all_channels(eta_counts, [0.0, 0.0])
        return (counts_distribution[0] +
                counts_distribution[1]) / (plays * eta_group_length)

    def _create_one_circuit_without_measurement(self,
                                                configuration: ChannelConfiguration,
                                                eta: float) -> Tuple[ClassicalRegister, QuantumCircuit]:
        """ Creates one circuit from a given  configuration and eta """
        configuration = cast(OneShotConfiguration, configuration)
        qreg_q = QuantumRegister(2, 'q')
        creg_c = ClassicalRegister(1, 'c')
        initial_state = self._prepare_initial_state(configuration.state_probability)
        circuit = QuantumCircuit(qreg_q, creg_c)
        circuit.initialize([initial_state[0],
                            initial_state[1]], qreg_q[0])
        circuit.reset(qreg_q[1])
        circuit.cry(2 * eta, qreg_q[0], qreg_q[1])
        circuit.cx(qreg_q[1], qreg_q[0])
        circuit.rx(configuration.angle_rx, qreg_q[0])
        circuit.ry(configuration.angle_ry, qreg_q[0])
        return creg_c, circuit

    def _add_measurement_to_one_circuit(self,
                                        creg_c: ClassicalRegister,
                                        circuit: QuantumCircuit) -> QuantumCircuit:
        circuit.measure([0], creg_c[0])
        return circuit

    def _create_one_configuration(self,
                                  configuration: ChannelConfiguration,
                                  eta_group: List[float]) -> OneShotConfiguration:
        """ Creates a specific configuration setting a specific eta group """
        return OneShotConfiguration({
            'state_probability': cast(OneShotConfiguration, configuration).state_probability,
            'angle_rx': cast(OneShotConfiguration, configuration).angle_rx,
            'angle_ry': cast(OneShotConfiguration, configuration).angle_ry,
            'eta_group': eta_group
        })

    def _compute_upper_and_lower_fidelity_bounds(self, state_vectors: List[np.ndarray]) -> Fidelities:
        """ Computes upper and lower fidelity bounds from the given state vectors """
        density_matrices = [partial_trace(np.outer(state_vector, state_vector.conj()), [1])
                            for state_vector in state_vectors]
        index_pairs = list(itertools.combinations(range(len(density_matrices)), 2))
        channel_fidelities = [fidelity(Qobj(density_matrices[first_index_pair].data), Qobj(
            density_matrices[second_index_pair].data)) for first_index_pair, second_index_pair in index_pairs]
        return Fidelities(upper_bound_fidelity=upper_bound_fidelity(channel_fidelities),
                          lower_bound_fidelity=lower_bound_fidelity(channel_fidelities))

    def _get_probabilities_and_etas_assigned_from_counts(self,
                                                         eta_counts: List[dict],
                                                         plays: int,
                                                         eta_group_length: int) -> ValidatedConfiguration:
        """ Computes the validated probability from the 'counts' measured
            based on the guess strategy that is required to use and returns the
            etas assigned for each measured state
        """
        eta_assignments = MeasuredStatesEtaAssignmentOneQubit(state_0=-1,
                                                              state_1=-1)
        mesaured_states_counts = MeasuredStatesCountsOneQubit(state_0=[],
                                                              state_1=[],
                                                              total_counts=(plays * eta_group_length))

        max_counts, eta_assignments, measured_states_counts = self._get_max_counts_and_etas_for_all_channels(
            eta_counts,
            eta_assignments,
            measured_states_counts=mesaured_states_counts)
        global_average_success_probability, etas_probability = self._get_global_and_etas_probabilities(
            max_counts, eta_assignments, plays, eta_group_length)

        return ValidatedConfiguration(validated_probability=global_average_success_probability,
                                      etas_probability=etas_probability,
                                      measured_states_eta_assignment=eta_assignments,
                                      measured_states_counts=measured_states_counts)

    def _get_max_counts_and_etas_for_all_channels(
            self,
            all_channel_counts: List[dict],
            eta_assignments: MeasuredStatesEtaAssignmentOneQubit,
            measured_states_counts: MeasuredStatesCountsOneQubit) -> Tuple[
                List[int],
                MeasuredStatesEtaAssignmentOneQubit,
                MeasuredStatesCountsOneQubit]:
        """ returns the max counts between the max counts up to that moment and the circuit counts
            assigning the winner eta channel
        """
        max_counts = [0, 0]
        for current_eta, one_channel_counts in enumerate(all_channel_counts):
            if '0' in one_channel_counts:
                max_counts[0], eta_assignments['state_0'] = self._update_max_counts_and_eta_assignment(
                    eta_assignments['state_0'],
                    max_counts[0],
                    one_channel_counts['0'],
                    current_eta)
                measured_states_counts['state_0'].append(one_channel_counts['0'])
            if '1' in one_channel_counts:
                max_counts[1], eta_assignments['state_1'] = self._update_max_counts_and_eta_assignment(
                    eta_assignments['state_1'],
                    max_counts[1],
                    one_channel_counts['1'],
                    current_eta)
                measured_states_counts['state_1'].append(one_channel_counts['1'])
            measured_states_counts = self._update_null_measured_counts(one_channel_counts, measured_states_counts)
        return (max_counts, eta_assignments, measured_states_counts)

    def _update_max_counts_and_eta_assignment(self,
                                              max_eta: int,
                                              max_counts: int,
                                              new_value: int,
                                              new_eta: int) -> Tuple[int, int]:
        """ return the max value between the new value or the current and
            increase the eta index when new value is greater or
            when is the same value, increase it on 50% of the times
        """
        if new_value > max_counts:
            return (new_value, new_eta)
        if new_value == max_counts:
            random_assignment = random.choice([max_eta, new_eta])
            return (new_value, random_assignment)
        return (max_counts, max_eta)

    def _update_null_measured_counts(
            self,
            one_channel_counts: dict,
            measured_states_counts: MeasuredStatesCountsOneQubit) -> MeasuredStatesCountsOneQubit:
        if '0' not in one_channel_counts:
            measured_states_counts['state_0'].append(0)
        if '1' not in one_channel_counts:
            measured_states_counts['state_1'].append(0)
        return measured_states_counts

    def _get_global_and_etas_probabilities(self,
                                           max_counts: List[int],
                                           etas_assignments: MeasuredStatesEtaAssignmentOneQubit,
                                           plays: int,
                                           eta_group_length: int) -> Tuple[float, List[float]]:
        """ return the global average success probability and the one from each eta channel """
        global_average_success_probability = (max_counts[0] +
                                              max_counts[1]) / (plays * eta_group_length)

        eta_counts = [0] * eta_group_length
        for idx, state in enumerate(etas_assignments):
            if state != 'state_0' and state != 'state_1':
                raise ValueError(f'invalid state: {state}')
            eta_assigned = etas_assignments[state]  # type: ignore
            if eta_assigned < -1 or eta_assigned > 1:
                raise ValueError(f'invalid eta: {eta_assigned}')
            if eta_assigned != -1:
                eta_counts[eta_assigned] += max_counts[idx]

        eta_probabilities = [eta_count / (plays * eta_group_length) for eta_count in eta_counts]
        if np.round(sum(eta_probabilities), 1) != np.round(global_average_success_probability, 1):
            raise ValueError(f'invalid probabilities! Globa avg: {global_average_success_probability} and ' +
                             f'sum probs: {np.round(sum(eta_probabilities), 2)}')
        return (global_average_success_probability, eta_probabilities)
