# https://github.com/francienbarkhof/Quantum_Group1_Project/blob/534e7d4b9c6805b7dcc4dae677869e1a563e391b/Noise_sim.py
import numpy as np
from qiskit import *
from qiskit_aer.noise import (NoiseModel, QuantumError, ReadoutError,
    pauli_error, depolarizing_error, thermal_relaxation_error)
aer_sim = Aer.get_backend('aer_simulator')

def Noise_sim_depol(p_meas,p_gate):
    error_meas = pauli_error([('X',p_meas), ('I', 1 - p_meas)])
    error_gate1 = depolarizing_error(p_gate, 1)
    error_gate2 = error_gate1.tensor(error_gate1)

    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(error_meas, "measure") # measurement error is applied to measurements
    noise_model.add_all_qubit_quantum_error(error_gate1, ["x"]) # single qubit gate error is applied to x gates
    noise_model.add_all_qubit_quantum_error(error_gate2, ["cx"]) # two qubit gate error is applied to cx gates
        
    return noise_model

def Noise_sim_thermal():
    # T1 and T2 values for qubits 0-3
    T1s = np.random.normal(50e3, 10e3, 4) # Sampled from normal distribution mean 50 microsec
    T2s = np.random.normal(70e3, 10e3, 4)  # Sampled from normal distribution mean 50 microsec

    # Truncate random T2s <= T1s
    T2s = np.array([min(T2s[j], 2 * T1s[j]) for j in range(4)])

    # Instruction times (in nanoseconds)
    time_u1 = 0   # virtual gate
    time_u2 = 50  # (single X90 pulse)
    time_u3 = 100 # (two X90 pulses)
    time_cx = 300
    time_reset = 1000  # 1 microsecond
    time_measure = 1000 # 1 microsecond

    # QuantumError objects
    errors_reset = [thermal_relaxation_error(t1, t2, time_reset)
                    for t1, t2 in zip(T1s, T2s)]
    errors_measure = [thermal_relaxation_error(t1, t2, time_measure)
                    for t1, t2 in zip(T1s, T2s)]
    errors_u1  = [thermal_relaxation_error(t1, t2, time_u1)
                    for t1, t2 in zip(T1s, T2s)]
    errors_u2  = [thermal_relaxation_error(t1, t2, time_u2)
                    for t1, t2 in zip(T1s, T2s)]
    errors_u3  = [thermal_relaxation_error(t1, t2, time_u3)
                    for t1, t2 in zip(T1s, T2s)]
    errors_cx = [[thermal_relaxation_error(t1a, t2a, time_cx).expand(
                    thermal_relaxation_error(t1b, t2b, time_cx))
    for t1a, t2a in zip(T1s, T2s)]
    for t1b, t2b in zip(T1s, T2s)]

    # Add errors to noise model
    noise_thermal = NoiseModel()
    noise_thermal.add_quantum_error(errors_reset[0], "reset", [0])
    noise_thermal.add_quantum_error(errors_measure[0], "measure", [0])
    noise_thermal.add_quantum_error(errors_u1[0], "u1", [0])
    noise_thermal.add_quantum_error(errors_u2[0], "u2", [0])
    noise_thermal.add_quantum_error(errors_u3[0], "u3", [0])
    noise_thermal.add_quantum_error(errors_reset[1], "reset", [1])
    noise_thermal.add_quantum_error(errors_measure[1], "measure", [1])
    noise_thermal.add_quantum_error(errors_u1[1], "u1", [1])
    noise_thermal.add_quantum_error(errors_u2[1], "u2", [1])
    noise_thermal.add_quantum_error(errors_u3[1], "u3", [1])
    noise_thermal.add_quantum_error(errors_reset[2], "reset", [2])
    noise_thermal.add_quantum_error(errors_measure[2], "measure", [2])
    noise_thermal.add_quantum_error(errors_u1[2], "u1", [2])
    noise_thermal.add_quantum_error(errors_u2[2], "u2", [2])
    noise_thermal.add_quantum_error(errors_u3[2], "u3", [2])
    noise_thermal.add_quantum_error(errors_reset[3], "reset", [3])
    noise_thermal.add_quantum_error(errors_measure[3], "measure", [3])
    noise_thermal.add_quantum_error(errors_u1[3], "u1", [3])
    noise_thermal.add_quantum_error(errors_u2[3], "u2", [3])
    noise_thermal.add_quantum_error(errors_u3[3], "u3", [3])
    noise_thermal.add_quantum_error(errors_cx[j][0], "cx", [j, 0])
    noise_thermal.add_quantum_error(errors_cx[j][1], "cx", [j, 1])
    noise_thermal.add_quantum_error(errors_cx[j][2], "cx", [j, 2])
    noise_thermal.add_quantum_error(errors_cx[j][3], "cx", [j, 3])

    return noise_thermal

print("finish")