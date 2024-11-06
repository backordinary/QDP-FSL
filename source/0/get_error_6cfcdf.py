# https://github.com/gwjacobson/QuantumErrorCorrection/blob/8ed2ad24cbae105221942fa1bd128285904aa9cc/get_error.py
from qiskit import *
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors import pauli_error, depolarizing_error

## Copied from QISKIT documentation. Tested out the noise model on a single qubit.

## Doesn't have great correlation with what I am trying to accomplish, since it uses
##qasm simulator and wouldn't allow me to track state fidelity,
##so, chose not to use it. But, I'll keep it in here. It's a cool function, otherwise.

def get_noise(p_meas,p_gate):
    error_meas = pauli_error([('X',p_meas), ('I', 1 - p_meas)])
    error_gate1 = depolarizing_error(p_gate, 1)
    error_gate2 = error_gate1.tensor(error_gate1)

    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(error_meas, "measure") # measurement error is applied to measurements
    noise_model.add_all_qubit_quantum_error(error_gate1, ["x"]) # single qubit gate error is applied to x gates
    noise_model.add_all_qubit_quantum_error(error_gate2, ["cx"]) # two qubit gate error is applied to cx gates
        
    return noise_model