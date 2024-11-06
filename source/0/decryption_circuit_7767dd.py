# https://github.com/OrenScheer/certified-deletion/blob/a84b1ca077a59705c55086dd9224f627829f0355/decryption_circuit.py
"""The circuit and associated methods that are used to decrypt a given ciphertext."""

from typing import List, Tuple, Dict
from qiskit import QuantumCircuit
from states import Basis, Key, Ciphertext
from encryption_circuit import calculate_error_correction_hash, calculate_privacy_amplification_hash
from utils import xor
from scheme_parameters import SchemeParameters


def create_decryption_circuit(key: Key, ciphertext: Ciphertext) -> List[QuantumCircuit]:
    """Creates and returns the decryption circuit, given a Ciphertext and an associated Key."""
    decryption_circuits = [circuit.copy() for circuit in ciphertext.circuits]
    qubit_count = 0
    for circuit in decryption_circuits:
        circuit.barrier()
        h_indices = [i for i in range(
            circuit.num_qubits) if key.theta[qubit_count + i] is Basis.HADAMARD]
        if h_indices:
            circuit.h(h_indices)
        circuit.measure_all()
        qubit_count += circuit.num_qubits
    return decryption_circuits


def decrypt_results(measurements: Dict[str, int], key: Key, ciphertext: Ciphertext, message: str, scheme_parameters: SchemeParameters, error_correct: bool = True) -> Tuple[int, int, int, int]:
    """Processes and decrypts the candidate decryption measurements for a sequence of experimental tests.

    Args:
        measurements: A dictionary whose keys are the measurements of all the qubits by the receiving
            party once the key is revealed, and whose values are the number of times that each measurement
            string has occurred experimentally.
        key: The key to be used in the decryption circuit.
        ciphertext: The ciphertext that the receiving party possesses.
        message: The original plaintext, to compare with the candidate decryption.
        scheme_parameters: The parameters of this instance of the BI20 scheme.
        error_correct: Whether or not to apply the error correction procedure.

    Returns:
        A tuple (correct_decryption_with_error_flag, correct_decryption_no_error_flag, incorrect_decryption_with_error_flag,
        incorrect_decryption_no_error_flag) where each value is the count of how many times that outcome occurred using
        the provided measurements.
    """
    correct_decryption_with_error_flag = 0
    correct_decryption_no_error_flag = 0
    incorrect_decryption_with_error_flag = 0
    incorrect_decryption_no_error_flag = 0
    for measurement, count in measurements.items():
        successful_decryption, error_flag = decrypt_single_result(
            measurement, key, ciphertext, message, scheme_parameters, error_correct)
        if successful_decryption and error_flag:
            correct_decryption_with_error_flag += count
        elif successful_decryption and not error_flag:
            correct_decryption_no_error_flag += count
        elif not successful_decryption and error_flag:
            incorrect_decryption_with_error_flag += count
        elif not successful_decryption and not error_flag:
            incorrect_decryption_no_error_flag += count
    return correct_decryption_with_error_flag, correct_decryption_no_error_flag, incorrect_decryption_with_error_flag, incorrect_decryption_no_error_flag


def decrypt_single_result(measurement: str, key: Key, ciphertext: Ciphertext, message: str, scheme_parameters: SchemeParameters, error_correct: bool) -> Tuple[bool, bool]:
    """Decrypts a single measurement from a decryption circuit.

    Args:
        measurement: A string which is either a measurement of all the qubits, or just the one-time pad qubits.
        key: The key to be used in the decryption circuit.
        ciphertext: The ciphertext that the receiving party possesses.
        message: The original plaintext, to compare with the candidate decryption.
        scheme_parameters: The parameters of this instance of the BI20 scheme.
        error_correct: Whether or not to apply the error correction procedure.

    Returns:
        A tuple of two boolean values (successful_decryption, error_flag), where successful_decryption
        is True if following the decryption procuedure resulted in recovering the correct plaintext,
        and where error_flag is True if the error correction hash did not match or the error-correcting
        code detected but could not fix errors.
    """
    if len(measurement) == key.theta.count(Basis.COMPUTATIONAL):
        # Only the computational basis qubits were measured
        relevant_bits = measurement
    else:
        # All the qubits were measured
        relevant_bits = "".join(
            [ch for i, ch in enumerate(measurement) if key.theta[i] is Basis.COMPUTATIONAL])
    if error_correct:
        relevant_bits, error_correction_succeeded = scheme_parameters.corr(
            relevant_bits, xor(ciphertext.q, key.e))
    else:
        error_correction_succeeded = False
    error_correction_hash = xor(calculate_error_correction_hash(
        key.error_correction_matrix, relevant_bits), key.d)
    error_flag = error_correction_hash != ciphertext.p or not error_correction_succeeded
    x_prime = calculate_privacy_amplification_hash(
        key.privacy_amplification_matrix, relevant_bits)
    decrypted_string = xor(ciphertext.c, x_prime, key.u)
    successful_decryption = decrypted_string == message
    return successful_decryption, error_flag
