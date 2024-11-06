# https://github.com/iamtxena/quantum-channel-discrimination/blob/20b63289fac85feda3d0cf3a40ff9816e8094626/qcd/dampingchannels/oneshotentangledfulluniversal.py
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qcd.configurations.configuration import ChannelConfiguration
from qcd.optimizationresults.aux import load_result_from_file, save_result_to_disk
from qcd.configurations import OneShotSetupConfiguration
from qcd.circuits import OneShotEntangledFullUniversalCircuit
from typing import List, Optional, Tuple
from . import OneShotEntangledUniversalDampingChannel
from ..typings import CloneSetup, OptimizationSetup, ResultStates
from ..optimizations import OneShotEntangledFullUniversalOptimization
from ..typings.configurations import OptimalConfigurations, ValidatedConfiguration
import numpy as np


class OneShotEntangledFullUniversalDampingChannel(OneShotEntangledUniversalDampingChannel):
    """ Representation of the One Shot Two Qubit Entangled with Full Input Quantum Damping Channel """

    @staticmethod
    def build_from_optimal_configurations(file_name: str, path: Optional[str] = ""):
        """ Builds a Quantum Damping Channel from the optimal configuration for each pair of attenuation angles """
        return OneShotEntangledFullUniversalDampingChannel(
            optimal_configurations=load_result_from_file(file_name, path))

    @staticmethod
    def find_optimal_configurations(optimization_setup: OptimizationSetup,
                                    clone_setup: Optional[CloneSetup] = None) -> OptimalConfigurations:
        """ Finds out the optimal configuration for each pair of attenuation levels
          using the configured optimization algorithm for an Entangled channel """

        optimal_configurations = OneShotEntangledFullUniversalOptimization(
            optimization_setup).find_optimal_configurations(clone_setup)
        if clone_setup is not None and clone_setup['file_name'] is not None:
            save_result_to_disk(optimal_configurations,
                                f"{clone_setup['file_name']}_{clone_setup['id_clone']}", clone_setup['path'])
        return optimal_configurations

    @staticmethod
    def discriminate_channel(configuration: ChannelConfiguration, plays: Optional[int] = 100) -> float:
        """ Computes the average success probability of running a specific configuration
            for the number of plays specified.
        """
        return OneShotEntangledFullUniversalCircuit().compute_average_success_probability(configuration, plays)

    @staticmethod
    def validate_optimal_configuration(configuration: ChannelConfiguration,
                                       plays: Optional[int] = 10000) -> ValidatedConfiguration:
        """ Runs the circuit with the given optimal configuration computing the success average probability
            for each eta (and also the global), the selected eta for each measured state and finally the
            upper and lower bound fidelities
        """
        return OneShotEntangledFullUniversalCircuit().validate_optimal_configuration(configuration, plays)

    def __init__(self,
                 channel_setup_configuration: Optional[OneShotSetupConfiguration] = None,
                 optimal_configurations: Optional[OptimalConfigurations] = None) -> None:
        super().__init__(channel_setup_configuration, optimal_configurations)
        if optimal_configurations is not None:
            self._one_shot_circuit = OneShotEntangledFullUniversalCircuit(optimal_configurations)

    def _create_all_circuits(self,
                             channel_setup_configuration: OneShotSetupConfiguration) -> Tuple[List[QuantumCircuit],
                                                                                              ResultStates]:
        qreg_q = QuantumRegister(3, 'q')
        creg_c = ClassicalRegister(2, 'c')

        circuits = []
        # Initialize circuit with desired initial_state
        initial_states = self._prepare_initial_states(
            channel_setup_configuration.angles_theta, channel_setup_configuration.angles_phase)

        for attenuation_factor in channel_setup_configuration.attenuation_factors:
            circuit_one_attenuation_factor = []
            circuit = QuantumCircuit(qreg_q, creg_c)
            circuit.u3(0, 0, 0, qreg_q[0])
            circuit.u3(0, 0, 0, qreg_q[1])
            circuit.cx(qreg_q[1], qreg_q[0])
            circuit.u3(0, 0, 0, qreg_q[0])
            circuit.u3(0, 0, 0, qreg_q[1])
            circuit.cx(qreg_q[0], qreg_q[1])
            circuit.u3(0, 0, 0, qreg_q[0])
            circuit.u3(0, 0, 0, qreg_q[1])
            circuit.cx(qreg_q[1], qreg_q[0])
            circuit.u3(0, 0, 0, qreg_q[0])
            circuit.u3(0, 0, 0, qreg_q[1])
            circuit.reset(qreg_q[2])
            circuit.barrier()
            circuit.cry(2 * np.arcsin(np.sqrt(attenuation_factor)), qreg_q[1], qreg_q[2])
            circuit.cx(qreg_q[2], qreg_q[1])
            circuit.barrier()
            circuit.u3(0, 0, 0, qreg_q[0])
            circuit.u3(0, 0, 0, qreg_q[1])
            circuit.cx(qreg_q[1], qreg_q[0])
            circuit.u3(0, 0, 0, qreg_q[0])
            circuit.u3(0, 0, 0, qreg_q[1])
            circuit.cx(qreg_q[0], qreg_q[1])
            circuit.u3(0, 0, 0, qreg_q[0])
            circuit.u3(0, 0, 0, qreg_q[1])
            circuit.cx(qreg_q[1], qreg_q[0])
            circuit.u3(0, 0, 0, qreg_q[0])
            circuit.u3(0, 0, 0, qreg_q[1])
            circuit.measure([0, 1], creg_c)
            circuit_one_attenuation_factor.append(circuit)
            circuits.append(circuit_one_attenuation_factor)
        return circuits, initial_states
