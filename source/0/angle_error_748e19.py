# https://github.com/dowusu-antwi/qiskit-tutorials/blob/01ac1b346839324b9ce682c7191c660630381d5f/angle_error.py
########################### QUESTIONS BEGIN WITH '?' ###########################

import numpy as np
import matplotlib.pyplot as plt

print("===============================================")
print("Importing qiskit modules...")
from qiskit import Aer, execute
from qiskit.providers.aer.noise.errors.standard_errors \
    import coherent_unitary_error
from qiskit.providers.aer.noise import NoiseModel

from qiskit.ignis.characterization.gates import (AngleCalFitter,
                                                 anglecal_1Q_circuits, 
                                                 AngleCalCXFitter,
                                                 anglecal_cx_circuits)
print("Import complete.")

def open_plot():
    '''
    Opens and maximizes the current figure.
    '''
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()


## Angle Error Characterization for CNOT Gates

# Measure angle error in CNOT gate wrt to angle of single qubit gates
qubits = [0, 2]
controls = [1, 3]
max_repetitions = 15
angle_error = 0.1
circuits, delay_times = anglecal_cx_circuits(max_repetitions, qubits, controls,
                                             angleerr=angle_error)


# Sets and runs simulator
backend = Aer.get_backend('qasm_simulator')
trials = 1000
result = execute(circuits, backend, shots=trials).result()


# Fits data to an oscillation
plt.figure(figsize=(10, 6))
theta, c = 0.02, 0.5
initial_parameter_bias = [theta, c]
lower_bounds, upper_bounds = [-np.pi, -1], [np.pi, 1]
angle_calibration_cnot_fit = AngleCalCXFitter(result, delay_times, qubits,
                                              fit_p0=initial_parameter_bias,
                                              fit_bounds=(lower_bounds,
                                                          upper_bounds))
angle_calibration_cnot_fit.plot(0, ax=plt.gca())
cnot_rotation_error = angle_calibration_cnot_fit.angle_err()[0]
print("Rotation Error on CX: %f rads" % cnot_rotation_error)
open_plot()

if __name__ == "__main__":
    pass
