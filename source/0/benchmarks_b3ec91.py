# https://github.com/sami-b95/random_circuits/blob/ef9dce33e7de3585b4002af1e9ad5c48786f8bf0/benchmarks.py
import numpy as np
from qiskit import Aer, ClassicalRegister, execute, QuantumCircuit

import twirl
import twirl_optimized


def xeb_circuit_benchmark(qc, n_samples=8192, moment=1):
    # Extract circuit unitary
    statevector = execute(qc, backend=Aer.get_backend("statevector_simulator")).result().get_statevector()
    # Sample strings from circuit
    qr = qc.qregs[0]
    cr = ClassicalRegister(len(qr))
    measure_qc = QuantumCircuit(qr, cr)
    measure_qc.measure(qr, cr)
    full_qc = qc + measure_qc
    counts = execute(full_qc, backend=Aer.get_backend("qasm_simulator"), shots=n_samples, seed_simulator=np.random.randint(2 ** 32)).result().get_counts(full_qc)
    probabilities = []
    for bits_string, count in counts.items():
        probabilities += [np.abs(statevector[int(bits_string, 2)]) ** 2] * count
    return 2 ** (qc.n_qubits * moment) * np.mean(np.array(probabilities) ** moment) - 1, probabilities

def xeb_benchmark(circuit_sampler, n_circuits, moment=1):
    benchmarks = []
    all_probabilities = []
    for trial in range(n_circuits):
        qc = circuit_sampler()
        benchmark, probabilities = xeb_circuit_benchmark(qc, n_samples=8192, moment=moment)
        benchmarks.append(benchmark)
        all_probabilities.extend(probabilities)
    return 1 + (np.mean(benchmarks) - (np.math.factorial(moment + 1) - 1)) / (np.math.factorial(moment + 1) - 1), \
        np.std(benchmarks) / (np.sqrt(n_circuits) * (np.math.factorial(moment + 1) - 1)), \
        all_probabilities

def twirl_matrix_coefficient_benchmark(circuit_sampler, n_tensor_factors, n_circuits, avoid_sign_problem=True):
    """ Choose indices i_1, ..., i_t, i_1', ..., i_t', j_1, ..., j_t, j_1', ..., j_t' and random and
        compute U_{i_1, i_1'} ... U_{i_t, t_t'} conj(U_{j_1, j_1'}) ... conj(U_{j_t, j_t'}) for U
        a unitary implemented by a quantum circuit sampled at random. Measure the closeness of the
        result to the one obtained with Haar-random U and properly rescale the result (following
        1809.06957v1, theorem 8.1, statement 1).
    
    Parameters
    ----------
    circuit_sampler: function
        A function sampling a random quantum circuit (from some ensemble) in Qiskit format
        (QuantumCircuit class).
    n_tensor_factors: int
        The number of tensor powers in Haar averages (the 't' parameter from the description).
    n_circuit: int
        Number of random circuits to sample.
    avoid_sign_problem: bool
        If set to True, enforce the constraints i_1 = j_1, ..., i_t = j_t, i_1' = j_1', ..., i_t' = j_t'
        so that all sampled U_{i_1, i_1'} ... U_{i_t, t_t'} conj(U_{j_1, j_1'}) ... conj(U_{j_t, j_t'})
        are real and one avoids the sign problem.
    
    Returns
    -------
    tuple
        The tuple contains:
            - 2 ** (number of qubits * t) * (theoretical mean - empirical mean), which can be regarded as a lower
            bound for epsilon such that the circuit ensemble forms an epsilon-approximate design.
            - The estimated error on the quantity above, estimated from the standard deviation of the samples
            over sqrt(number of samples).
            - The list of randomly generated i_1, ..., i_t.
            - The list of randomly generated j_1, ..., j_t.
            - The list of randomly generated j_1', ..., j_t'.
            - The list of randomly generated i_1', ..., i_t'.
            - The generated samples.
    """
    n_qubits = circuit_sampler().n_qubits
    d_value = 2 ** n_qubits
    basis_labels_right = np.random.randint(0, d_value, n_tensor_factors)
    basis_labels_mid_right = np.random.randint(0, d_value, n_tensor_factors)
    basis_labels_mid_left = np.copy(basis_labels_mid_right)
    basis_labels_left = np.copy(basis_labels_right)
    theoretical_mean = twirl_optimized.numerical_haar_average_computational_basis_optimized(
        basis_labels_left,
        basis_labels_mid_left,
        basis_labels_mid_right,
        basis_labels_right,
        d_value
    )
    samples = twirl.circuit_statistical_sample_computational_basis(
        basis_labels_left,
        basis_labels_mid_left,
        basis_labels_mid_right,
        basis_labels_right,
        d_value,
        n_circuits,
        circuit_sampler
    )
    return 2 ** (n_qubits * n_tensor_factors) * np.abs(theoretical_mean - np.mean(samples)), \
        2 ** (n_qubits * n_tensor_factors) * np.std(samples) / np.sqrt(n_circuits), \
        basis_labels_left, \
        basis_labels_mid_left, \
        basis_labels_mid_right, \
        basis_labels_right, \
        samples

def anticoncentration_benchmark(circuit_sampler, n_circuits):
    n_qubits = circuit_sampler().n_qubits
    d_value = 2 ** n_qubits
    samples = twirl.circuit_statistical_sample_computational_basis(
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        d_value,
        n_circuits,
        circuit_sampler
    )
    return 2 ** n_qubits * (2 ** n_qubits + 1) * np.mean(samples).real - 1, \
        2 ** n_qubits * (2 ** n_qubits + 1) * np.std(samples) / np.sqrt(n_circuits)