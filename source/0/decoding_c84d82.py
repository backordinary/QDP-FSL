# https://github.com/tom-88/qhack22_qec_testbed/blob/06310a6a27ea61a68a0ea52da7a9fceded9d1e0a/decoding.py
from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister, execute, IBMQ, transpile, Aer
from qiskit.providers.aer import AerSimulator
from qiskit.quantum_info import random_statevector, Statevector, state_fidelity
from qiskit.opflow import I, X, Y, Z, StateFn
import numpy as np
from qiskit.tools.monitor import job_monitor
from qiskit.providers.aer.noise import NoiseModel, depolarizing_error
from scipy.linalg import eig, LinAlgError

from decoding_utils import *

statevector = Aer.get_backend('statevector_simulator')
qasm = Aer.get_backend('qasm_simulator')

# Parameters
num_qubits = 5
random_statevector = random_statevector(2,0)
shots = 16384 # this needs to be high for the encoded circuit expectations to have sufficiently low error to reproduce exact in noise free case

# TODO: for ease doing this when the observable is a single pauli term, need to to extend to sum of pauli strings, will be same form as the code space Hamiltonian conjugation 
target_observable = Z # the logical z
logical_target_observable = Z^Z^Z^Z^Z # TODO: need to do this algorithmically for an arbitrary target observable.

generators = [X^Z^Z^X^I, I^X^Z^Z^X, X^I^X^Z^Z, Z^X^I^X^Z]

num_generators = len(generators)
overlap_operators = [[(~m_i) @ m_j for m_j in generators] for m_i in generators]
code_space_hamiltonian = -1*sum(generators)
transformed_hamiltonian_operators = [[[-1 * ((~m_i) @ g @ m_j) for g in generators] for m_j in generators] for m_i in generators] # defer the summing of the conjugated code space Hamiltonian until later
corrected_observable_operators = [[(~m_i) @ logical_target_observable @ m_j for m_j in generators] for m_i in generators]


# Calculate exact expectation for comparison.
exact_expectation = np.abs(StateFn(target_observable).adjoint().eval(StateFn(random_statevector)))

depolarising_probabilities = [i / 100 for i in range(0,51,5)]
unencoded_expectations = []
uncorrected_encoded_expectations = []
corrected_encoded_expectations = []

for p in depolarising_probabilities:
    # Use parameterised noise model for ease of control
    noise_model = NoiseModel()

    error = depolarizing_error(p, 1)
    noise_model.add_all_qubit_quantum_error(error, ['u1', 'u2', 'u3'])

    backend = AerSimulator(noise_model=noise_model)
    
    # Unencoded circuit with noise
    qr = QuantumRegister(1)
    unencoded_qc = QuantumCircuit(qr)
    unencoded_qc.initialize(random_statevector.data, 0)
    unencoded_qc = hadamard_test(1, unencoded_qc, target_observable.to_circuit())
    unencoded_target_observable_counts = execute(unencoded_qc, backend, shots=shots).result().get_counts()
    unencoded_expectations.append(np.abs((unencoded_target_observable_counts.get('0', 0) - unencoded_target_observable_counts.get('1', 0)) / shots))
    
    # Encoded circuit with noise and no correct operations
    encoded_uncorr_qc = encode_5_qubit(random_statevector.data)   
    encoded_uncorr_qc = hadamard_test(num_qubits, encoded_uncorr_qc, logical_target_observable.to_circuit())
    encoded_uncorr_target_observable_counts = execute(encoded_uncorr_qc, backend, shots=shots).result().get_counts()
    uncorrected_encoded_expectations.append(np.abs((encoded_uncorr_target_observable_counts.get('0', 0) - encoded_uncorr_target_observable_counts.get('1', 0)) / shots))

    overlap = np.zeros((num_generators, num_generators))
    transformed_hamiltonian = np.zeros((num_generators, num_generators))
    corrected_observable_expectations = np.zeros((num_generators, num_generators))
    
    # Estimate required expectation values using quantum computer
    for i in range(num_generators):
       for j in range(num_generators): 
            corrected_observable_qc = encode_5_qubit(random_statevector.data)
            corrected_observable_qc = hadamard_test(num_qubits, corrected_observable_qc, corrected_observable_operators[i][j].to_circuit())
            corrected_observable_counts = execute(corrected_observable_qc, backend, shots=shots).result().get_counts()
            corrected_observable_expectations[i][j] = np.abs((corrected_observable_counts.get('0', 0) - corrected_observable_counts.get('1', 0)) / shots)


            overlap_qc = encode_5_qubit(random_statevector.data)
            overlap_qc = hadamard_test(num_qubits, overlap_qc, overlap_operators[i][j].to_circuit())
            overlap_counts = execute(overlap_qc, backend, shots=shots).result().get_counts()
            overlap[i][j] = np.abs((overlap_counts.get('0', 0) - overlap_counts.get('1', 0)) / shots)

           
            for pauli_term in transformed_hamiltonian_operators[i][j]:
                transformed_ham_exp_qc = encode_5_qubit(random_statevector.data)
                transformed_ham_counts = execute(hadamard_test(num_qubits, transformed_ham_exp_qc, pauli_term.to_circuit()), backend, shots=shots).result().get_counts()
                transformed_hamiltonian[i][j] += np.abs((transformed_ham_counts.get('0', 0) - transformed_ham_counts.get('1', 0)) / shots)
     
    # Solve classical eigenvalue problem
    try:
        evals, evecs = eig(transformed_hamiltonian, overlap, right=True)
    except LinAlgError:
        print('scipy.eig eigenvalue computation did not converge.')
        quit()

    # With the optimal coefficients for the subspace expansion (observe with and without diff operators) show you can correct some observable e.g. fidelity. 
    min_eval_index = [i for i in range(len(evals)) if evals[i] == min(evals)][0]
    min_evec = [row[min_eval_index] for row in evecs]

    normalisation_factor = sum([sum([coeff_i * np.conj(coeff_j) * element for (coeff_j, element) in zip (min_evec, row)]) for (coeff_i, row) in zip(min_evec, overlap)])

    # Compute corrected observable with normalisation factor and optimal check operator weights
    corrected_observable = np.abs(sum([sum([coeff_i * np.conj(coeff_j) * element for (coeff_j, element) in zip (min_evec, row)]) for (coeff_i, row) in zip(min_evec, corrected_observable_expectations)]) / normalisation_factor)

    corrected_encoded_expectations.append(corrected_observable)

# Visualisation of results

import matplotlib.pyplot as plt

plt.plot(depolarising_probabilities, [exact_expectation for i in range(len(depolarising_probabilities))], '--', label='exact')
plt.plot(depolarising_probabilities, unencoded_expectations, 'x', label='unencoded', markersize=12)
plt.plot(depolarising_probabilities, corrected_encoded_expectations, '+', label='encoded & corrected', markersize=12)
plt.plot(depolarising_probabilities, uncorrected_encoded_expectations, '*', label='encoded', markersize=12)
plt.xlabel('Depolarising Probability')
plt.ylabel('Expectation of target observable')
plt.legend(loc='best')
plt.show()

plt.plot(depolarising_probabilities, [np.abs(exp - exact_expectation) / exact_expectation for exp in unencoded_expectations], 'x', label='unencoded', markersize=12)
plt.plot(depolarising_probabilities, [np.abs(exp - exact_expectation) / exact_expectation for exp in corrected_encoded_expectations], '+', label='encoded & corrected', markersize=12)
plt.plot(depolarising_probabilities,[np.abs(exp - exact_expectation) / exact_expectation for exp in uncorrected_encoded_expectations], '*', label='encoded', markersize=12)
plt.xlabel('Depolarising Probability')
plt.ylabel('Absolute error of expectation of target observable')
plt.legend(loc='best')
plt.show()

