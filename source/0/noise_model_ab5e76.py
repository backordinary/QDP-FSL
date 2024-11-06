# https://github.com/bfedrici-phd/QC-2020-CPE/blob/9f5788ecf186306799fd67a4a74f2f09df9c9591/Assignements/Project_%231/noise_model.py
import numpy as np

import qiskit
from qiskit.providers.aer.noise.errors.standard_errors import thermal_relaxation_error
from qiskit.providers.aer.noise import NoiseModel


def t1_noise_model(gate_time=0.1, t=[70.5, 85.0, 80.0, 90.5, 77.5]):
    """
    Return a NoiseModel object for T1.
    
    Parameters
        - gate_time: gate time (in microseconds) for a single-qubit gate
        - t: simulated times (in microseconds) for a set of five qubits 
    """
    
    t1_noise_model = NoiseModel()
    error = [0 for i in range(5)]
    error[0] = thermal_relaxation_error(t[0], 2*t[0], gate_time)
    t1_noise_model.add_quantum_error(error[0], 'id', [0])
    error[1] = thermal_relaxation_error(t[1], 2*t[1], gate_time)
    t1_noise_model.add_quantum_error(error[1], 'id', [1])
    error[2] = thermal_relaxation_error(t[2], 2*t[2], gate_time)
    t1_noise_model.add_quantum_error(error[2], 'id', [2])
    error[3] = thermal_relaxation_error(t[3], 2*t[3], gate_time)
    t1_noise_model.add_quantum_error(error[3], 'id', [3])
    error[4] = thermal_relaxation_error(t[4], 2*t[4], gate_time)
    t1_noise_model.add_quantum_error(error[4], 'id', [4])
    
    return t1_noise_model


def t2_star_noise_model(gate_time=0.1, t=[70.5, 85.0, 80.0, 90.5, 77.5]):
    """
    Return a NoiseModel object for T2*.
    
    Parameters
        - gate_time: gate time (in microseconds) for a single-qubit gate
        - t: simulated times (in microseconds) for a set of five qubits 
    """

    t2_star_noise_model = NoiseModel()
    error = [0 for i in range(5)]
    error[0] = thermal_relaxation_error(np.inf, t[0], gate_time, 0.5)
    t2_star_noise_model.add_quantum_error(error[0], 'id', [0])
    error[1] = thermal_relaxation_error(np.inf, t[1], gate_time, 0.5)
    t2_star_noise_model.add_quantum_error(error[1], 'id', [1])
    error[2] = thermal_relaxation_error(np.inf, t[2], gate_time, 0.5)
    t2_star_noise_model.add_quantum_error(error[2], 'id', [2])
    error[3] = thermal_relaxation_error(np.inf, t[3], gate_time, 0.5)
    t2_star_noise_model.add_quantum_error(error[3], 'id', [3])
    error[4] = thermal_relaxation_error(np.inf, t[4], gate_time, 0.5)
    t2_star_noise_model.add_quantum_error(error[4], 'id', [4])
    
    return t2_star_noise_model
