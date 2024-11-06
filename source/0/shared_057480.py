# https://github.com/aleksey-uvarov/verify-ghz/blob/89fae4fc0ec858afef70b224315865eeed94e6e5/shared.py
import numpy as np
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.providers.aer import AerSimulator
from qiskit.providers.aer.noise import NoiseModel, depolarizing_error


def sample_binomials(probabilities: np.array, total_shots):
    """Take the binomial success rates and produce
    total_shots samples, then return resampled probabilities."""
    shots_per_point = total_shots / probabilities.shape[0]
    resampled_frequencies = np.zeros_like(probabilities)
    for i, p in enumerate(probabilities):
        resampled_frequencies[i] = np.random.binomial(shots_per_point, p) / shots_per_point
    return resampled_frequencies


def get_ghz_circuit(n_qubits):
    q, c = QuantumRegister(n_qubits), ClassicalRegister(n_qubits)
    circ_ghz = QuantumCircuit(q, c)
    circ_ghz.h(q[0])
    for i in range(n_qubits - 1):
        circ_ghz.cx(q[i], q[i+1])
    return circ_ghz


def append_measurements_to_circ(circ, which):
    if which not in ('z', 'x'):
        raise ValueError("which must be 'z' or 'x'")
    q, c = circ.qregs[0], circ.cregs[0]
    circ_measure = circ.compose(QuantumCircuit(q, c))
    if which == 'x':
        for i in range(len(q)):
            circ_measure.h(q[i])
    circ_measure.measure(q, c)
    return circ_measure


def make_parametrized_cosine(n_qubits):

    def parametrized_cosine(x: np.array, amp: float, phase: float):
        return amp * np.cos(n_qubits * x - phase)

    return parametrized_cosine


def depolarizing_noise_model(p_single, p_cx):
    noise_model = NoiseModel()
    # Add depolarizing error to all single qubit u1, u2, u3 gates
    error = depolarizing_error(p_single, 1)
    error_cx = depolarizing_error(p_cx, 2)
    noise_model.add_all_qubit_quantum_error(error, ['u1', 'u2', 'u3', 'h'])
    noise_model.add_all_qubit_quantum_error(error_cx, ['cx'])
    return noise_model


def only_h_noise_model(p_h):
    """Noise model in which all gates except the Hadamard are perfect."""
    noise_model = NoiseModel()
    # Add depolarizing error to all single qubit u1, u2, u3 gates
    error = depolarizing_error(p_h, 1)
    noise_model.add_all_qubit_quantum_error(error, ['h'])
    return noise_model


def fidelity_population(n_qubits, n_shots, noise_model):
    backend = AerSimulator(noise_model=noise_model)
    circ = append_measurements_to_circ(get_ghz_circuit(n_qubits), 'z')
    result = backend.run(circ, shots=n_shots, seed=100).result()
    counts = result.get_counts(circ)

    # following the naming in Omran et al. (2019)
    alphas = (counts['0' * n_qubits] + counts['1' * n_qubits]) / n_shots
    alphas_variance = 1 - alphas**2
    return alphas, (alphas_variance / n_shots)**0.5  # again maybe wilson?