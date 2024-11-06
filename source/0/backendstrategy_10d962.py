# https://github.com/magn5452/QiskitQaoa/blob/c1ee63b4fc3bad2e2e31814386db6e564ae5aaf4/VehicleRouting/standard/qaoa/BackendStrategy.py
from qiskit import Aer
from qiskit.providers.aer import noise, StatevectorSimulator

from VehicleRouting.framework.qaoa.BackendStrategy import BackendStrategy


class AerBackendStrategy(BackendStrategy):
    def __init__(self):
        pass

    def get_backend(self):
        backend = Aer.get_backend('aer_simulator')
        backend.shots = 2^8
        return backend


class StateVectorBackendStrategy(BackendStrategy):
    def __init__(self):
        pass

    def get_backend(self):
        backend = StatevectorSimulator(precision='double')
        return backend


class NoisyBackendStrategy(BackendStrategy):
    def __init__(self):
        pass

    def get_backend(self):
        # Error probabilities
        prob_1 = 0.9  # 1-qubit gate
        prob_2 = 0.9  # 2-qubit gate

        # Depolarizing quantum errors
        error_1 = noise.depolarizing_error(prob_1, 1)
        error_2 = noise.depolarizing_error(prob_2, 2)

        # Add errors to noise model
        noise_model = noise.NoiseModel()
        noise_model.add_all_qubit_quantum_error(error_1, ['x', 'y', 'z', 'sx', 'h'])
        noise_model.add_all_qubit_quantum_error(error_2, ['cx', 'cy', 'cz'])
        backend = Aer.get_backend('aer_simulator', noise_model=noise_model)
        backend.shots = 1
        return backend
