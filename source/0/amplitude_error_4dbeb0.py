# https://github.com/dowusu-antwi/qiskit-tutorials/blob/01ac1b346839324b9ce682c7191c660630381d5f/amplitude_error.py
########################### QUESTIONS BEGIN WITH '?' ###########################

import numpy as np
import matplotlib.pyplot as plt

print("===============================================")
print("Importing qiskit modules...")
from qiskit import Aer, execute
from qiskit.providers.aer.noise.errors.standard_errors \
    import coherent_unitary_error
from qiskit.providers.aer.noise import NoiseModel

from qiskit.ignis.characterization.gates import (AmpCalFitter,
                                                 ampcal_1Q_circuits,
                                                 AngleCalFitter,
                                                 anglecal_1Q_circuits,
                                                 AmpCalCXFitter,
                                                 ampcal_cx_circuits,
                                                 AngleCalCXFitter,
                                                 anglecal_cx_circuits)
print("Import complete.")

def open_plot():
    '''
    Opens and maximizes the current figure.
    '''
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()


## Ammplitude Error Characterization for Single Qubit Gates

# Measures amplitude error, in the pi/2 pulse, on qubits 2 and 4
qubits = [4, 2]
max_circuit_repetitions = 10
circuits, delay_times = ampcal_1Q_circuits(max_circuit_repetitions, qubits)

# Sets simulator and adds rotation error.
angle_error = 0.1
error_unitary_matrix = np.zeros([2,2], dtype=complex)
error_unitary_matrix[0, 0] = np.cos(angle_error)
error_unitary_matrix[0, (0+1) % 2] = np.sin(angle_error)
error_unitary_matrix[1, 1] = np.cos(angle_error)
error_unitary_matrix[1, (1+1) % 2] = np.sin(angle_error)
error_unitary_matrix[0, 1] *= -1.0

error = coherent_unitary_error(error_unitary_matrix)
my_noise_model = NoiseModel()
my_noise_model.add_all_qubit_quantum_error(error, 'u2')

# Runs simulator
backend = Aer.get_backend('qasm_simulator')
trials = 500
result = execute(circuits, backend, shots=trials,
                 noise_model=my_noise_model).result()

# FIts data to an oscillation
plt.figure(figsize=(10,6))
theta, c = 0.02, 0.5                # ?: again, what are these parameter values?
initial_parameter_bias = [theta, c]
parameter_lower_bounds = [-np.pi, -1]
parameter_upper_bounds = [np.pi, 1]
amplitude_calibration_fit = AmpCalFitter(result, delay_times, qubits,
                                         fit_p0 = initial_parameter_bias,
                                         fit_bounds=(parameter_lower_bounds,
                                                     parameter_upper_bounds))
amplitude_calibration_fit.plot(1, ax=plt.gca())
open_plot()

rotation_error = amplitude_calibration_fit.angle_err()[0]
print("Rotation Error on U2: %f rads" % rotation_error)

if __name__ == "__main__":
    pass
