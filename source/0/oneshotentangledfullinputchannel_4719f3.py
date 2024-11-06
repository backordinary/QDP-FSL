# https://github.com/iamtxena/quantum-channel-discrimination/blob/20b63289fac85feda3d0cf3a40ff9816e8094626/qcd/dampingchannels/oneshotentangledfullinputchannel.py
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qcd.configurations.configuration import ChannelConfiguration
from qcd.optimizationresults.aux import load_result_from_file, save_result_to_disk
from qcd.configurations import OneShotSetupConfiguration
from qcd.circuits import OneShotEntangledFullInputCircuit
from typing import List, Optional, Tuple
from . import OneShotEntangledDampingChannel
from ..typings import CloneSetup, OptimizationSetup, ResultStates
from ..optimizations import OneShotEntangledFullInputOptimization
from ..typings.configurations import OptimalConfigurations
import numpy as np


class OneShotEntangledFullInputDampingChannel(OneShotEntangledDampingChannel):
    """ Representation of the One Shot Two Qubit Entangled with Full Input Quantum Damping Channel """

    @staticmethod
    def build_from_optimal_configurations(file_name: str, path: Optional[str] = ""):
        """ Builds a Quantum Damping Channel from the optimal configuration for each pair of attenuation angles """
        return OneShotEntangledFullInputDampingChannel(optimal_configurations=load_result_from_file(file_name, path))

    @staticmethod
    def find_optimal_configurations(optimization_setup: OptimizationSetup,
                                    clone_setup: Optional[CloneSetup] = None) -> OptimalConfigurations:
        """ Finds out the optimal configuration for each pair of attenuation levels
          using the configured optimization algorithm for an Entangled channel """

        optimal_configurations = OneShotEntangledFullInputOptimization(
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
        return OneShotEntangledFullInputCircuit().compute_average_success_probability(configuration, plays)

    def __init__(self,
                 channel_setup_configuration: Optional[OneShotSetupConfiguration] = None,
                 optimal_configurations: Optional[OptimalConfigurations] = None) -> None:
        super().__init__(channel_setup_configuration, optimal_configurations)
        if optimal_configurations is not None:
            self._one_shot_circuit = OneShotEntangledFullInputCircuit(optimal_configurations)

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
            circuit.rx(0, qreg_q[1])
            circuit.ry(0, qreg_q[1])
            circuit.rx(0, qreg_q[0])
            circuit.ry(0, qreg_q[0])
            circuit.cx(qreg_q[0], qreg_q[1])
            circuit.reset(qreg_q[2])
            circuit.cry(2 * np.arcsin(np.sqrt(attenuation_factor)), qreg_q[1], qreg_q[2])
            circuit.cx(qreg_q[2], qreg_q[1])
            circuit.cx(qreg_q[0], qreg_q[1])
            circuit.rx(0, qreg_q[1])
            circuit.ry(0, qreg_q[1])
            circuit.rx(0, qreg_q[0])
            circuit.ry(0, qreg_q[0])
            circuit.measure([0, 1], creg_c)
            circuit_one_attenuation_factor.append(circuit)
            circuits.append(circuit_one_attenuation_factor)
        return circuits, initial_states
