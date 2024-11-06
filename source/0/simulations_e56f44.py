# https://github.com/carloca/2021_02_qosf/blob/1597cd7c1adc436ce2e5adb24a1ebbb3eb7a9686/src/simulations.py
import random

from collections import defaultdict
from qiskit import Aer, execute, QuantumCircuit
from typing import Any, Dict, List, Tuple, Union

from src.bell_circuit import BellCircuit


def get_circuit_to_print(n_ancillas: int, correction_type: str) -> QuantumCircuit:
    """
    Generate circuit with symbolic error gates to be plot.

    :param n_ancillas: number of ancillas to correct the error.
    :param correction_type: type of error correction circuit.
                            Can be ['no_correction', 'simple', 'repetition_simple', 'shor']
    :return: Qiskit circuit with symbolic error gates.
    """
    qc = BellCircuit(n_ancillas, n_ancillas)
    if correction_type == 'simple':
        qc.create_circuit_with_simple_correction([], symb_err=True)
    elif correction_type == 'repetition_simple':
        qc.create_circuit_with_simple_repetition([], symb_err=True)
    elif correction_type == 'shor':
        qc.create_circuit_with_shor_correction([], symb_err=True)
    else:
        qc.create_circuit_with_no_correction([], symb_err=True)

    return qc.circuit


def run_one_combination(n_ancillas: int,
                        correction_type: str,
                        errors: List[Tuple[Tuple[int, int], str]]) -> Tuple[Dict[str, int], str]:
    """
    Run one error combination.

    :param n_ancillas: number of ancillas to correct the error.
    :param correction_type: type of error correction circuit.
                            Can be ['no_correction', 'simple', 'repetition_simple', 'shor']
    :param errors: Error set up to add to the circuit.
    :return: Measurements count and error in a string format.
    """

    backend = Aer.get_backend('qasm_simulator')
    qc = BellCircuit(n_ancillas, n_ancillas)
    error = f'[({errors[0][0][1]}, {errors[0][1]}), ({errors[1][0][1]}, {errors[1][1]})]'
    if correction_type == 'simple':
        # In this type of circuit we assume that the error cannot be in the ancillas, so we avoid those cases.
        error = f'[({0}, {errors[0][1]}), ({0}, {errors[1][1]})]'
        qc.create_circuit_with_simple_correction([((1, 0), errors[0][1]), ((2, 0), errors[1][1])])
    elif correction_type == 'repetition_simple':
        qc.create_circuit_with_simple_repetition(errors)
    elif correction_type == 'shor':
        qc.create_circuit_with_shor_correction(errors)
    else:
        qc.create_circuit_with_no_correction(errors)

    counts = execute(qc.circuit, backend, shots=1000).result().get_counts()

    return counts, error


def run_random_errors(n_ancillas: int,
                      correction_type: str,
                      iterations: int,
                      probabilities: List[Union[int, float]],
                      seed: Any) -> Tuple[Dict[str, int], Dict[str, Dict[str, int]]]:
    """
    Run multiple simulations of the circuit with random error setup.

    :param n_ancillas: number of ancillas to correct the error.
    :param correction_type: type of error correction circuit.
                            Can be ['no_correction', 'simple', 'repetition_simple', 'shor']
    :param iterations: number of iterations.
    :param probabilities: array with probabilities per each gate.
    :param seed: seed for random selection.
    :return: Total measurements count and error to count dictionary.
    """
    errors = ['i', 'x', 'z']
    all_counts = defaultdict(int)
    err_to_counts = {}
    random.seed(seed)

    for it in range(iterations):
        # Generate random error setup
        qubit_group_1 = random.randint(0, n_ancillas)
        qubit_group_2 = random.randint(0, n_ancillas)
        err_1 = random.choices(errors, weights=probabilities)
        err_2 = random.choices(errors, weights=probabilities)

        counts, error = run_one_combination(n_ancillas, correction_type, [((1, qubit_group_1), err_1[0]),
                                                                          ((2, qubit_group_2), err_2[0])])

        for key, value in counts.items():
            all_counts[key] += value

        if error in err_to_counts:
            for key, value in counts.items():
                if key in err_to_counts[error]:
                    err_to_counts[error][key] += value
                else:
                    err_to_counts[error][key] = value
        else:
            err_to_counts[error] = counts

    return all_counts, err_to_counts


def run_all_combinations(n_ancillas: int, correction_type: str) -> Tuple[Dict[str, int], Dict[str, Dict[str, int]]]:
    """
    Run all the possible error set up.

    :param n_ancillas: number of ancillas to correct the error.
    :param correction_type: type of error correction circuit.
                            Can be ['no_correction', 'simple', 'repetition_simple', 'shor']
    :return: Total measurements count and error to count dictionary.
    """
    errors = ['i', 'x', 'z']
    all_counts = defaultdict(int)
    err_to_counts = {}

    for qubit_group_1 in range(n_ancillas + 1):
        for qubit_group_2 in range(n_ancillas + 1):
            for err_1 in errors:
                for err_2 in errors:
                    counts, error = run_one_combination(n_ancillas, correction_type, [((1, qubit_group_1), err_1),
                                                                                      ((2, qubit_group_2), err_2)])
                    if error in err_to_counts:
                        # To avoid repetitions in case of simple error correction.
                        continue

                    for key, value in counts.items():
                        all_counts[key] += value
                    err_to_counts[error] = counts

    return all_counts, err_to_counts
