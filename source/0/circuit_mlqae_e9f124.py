# https://github.com/adamcallison/qae2/blob/9138119f89e971f33e72835ce4644fcdffe95b8e/circuit_mlqae.py
import numpy as np 
import circuit_util
from qiskit.providers.aer import AerSimulator
from qiskit import transpile
import mlqae
import warnings

import qiskit_aer.noise as noise

def qae(A, O, measure_qubits, results_to_good_count_func, max_grover_depth, \
    eps, delta, Nshot=None, shot_multiplier=None, jittigate=False, \
    noise_rate=None, kappa_params=None, compile_to=None):
    return mlqae.qae('circuit', \
        (A, O, measure_qubits, results_to_good_count_func,), qae_zerocounts, \
        max_grover_depth, eps, delta, Nshot=Nshot, \
        shot_multiplier=shot_multiplier, jittigate=jittigate, \
        noise_rate=noise_rate, kappa_params=kappa_params, compile_to=compile_to)

def qae_zerocounts(arg_tuple, grover_depths, shots_array, noise_rate=None, \
    compile_to=None, **kwargs):

    shots = np.max(shots_array) # due to qiskit limitations,
                                # warning raised elsewhere

    A, O, measure_qubits, results_to_good_count_func = arg_tuple

    total_shots, total_calls = 0, 0
    zeros = []

    circuits = circuit_util.qae_circuits(A, O, grover_depths, measure_qubits, \
        compile_to=compile_to)

    simulator = AerSimulator()
    if not (noise_rate is None):
        error = noise.depolarizing_error(noise_rate, 2)
        noise_model = noise.NoiseModel()
        noise_model.add_all_qubit_quantum_error(error, ['cx'])
        job = simulator.run(circuits, shots=shots, noise_model=noise_model)
    else:
        job = simulator.run(circuits, shots=shots)

    for j, grover_depth in enumerate(grover_depths):
        D = (2*grover_depth) + 1
        total_shots += shots
        total_calls += D*shots

        circuit = circuits[j]
        result = job.result()
        counts_circuit = result.get_counts(circuit)
        zerosd = results_to_good_count_func(counts_circuit)

        zeros.append(zerosd)
    zeros = np.array(zeros)
    return zeros, total_shots, total_calls
