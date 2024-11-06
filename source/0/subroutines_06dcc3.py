# https://github.com/Unathi-Skosana/prototype-variational-quantum-state-diagonalization/blob/a9a1036a3f37c9dacf1864665648e99e58918595/vqsd/utils/subroutines.py
"""
TODO
"""

from typing import Union, Iterable

import inspect
import warnings
import functools

import numpy as np


from qiskit.algorithms import optimizers
from qiskit.algorithms.optimizers import SPSA
from qiskit.algorithms.optimizers.spsa import powerseries

from qiskit import QuantumCircuit
from qiskit.result import Result
from vqsd.utils.logger import Logger as logger


def prepare_circuits_to_execute(
    stateprep_circuit: QuantumCircuit,
    ansatz: QuantumCircuit,
    params: Iterable[float],
    statevector_mode: bool = False,
):
    # pylint: disable=too-many-locals
    """
    TODO
    """
    circuits_to_execute = []

    num_qubits = stateprep_circuit.num_qubits // 2
    param_bindings = dict(zip(ansatz.parameters, params))
    u_circuit = ansatz.bind_parameters(param_bindings)

    first_copy_idx = np.arange(num_qubits, dtype=int).tolist()
    second_copy_idx = np.arange(num_qubits, 2 * num_qubits, dtype=int).tolist()

    dip_circuit_name_prefix = "dip_test" + "_" + str(stateprep_circuit.name) + "_0"
    pdip_circuit_name_prefix = "pdip_test" + "_" + str(stateprep_circuit.name) + "_"

    dip_circuit = QuantumCircuit(
        2 * num_qubits, num_qubits, name=dip_circuit_name_prefix
    )
    dip_circuit.compose(u_circuit, qubits=first_copy_idx, inplace=True)
    dip_circuit.compose(u_circuit, qubits=second_copy_idx, inplace=True)

    for qb_idx in first_copy_idx:
        dip_circuit.cx(qb_idx + num_qubits, qb_idx)

        if not statevector_mode:
            dip_circuit.measure(qb_idx, qb_idx)

    circuits_to_execute += [dip_circuit]

    pdip_circuit = dip_circuit.copy()

    for qb_idx in first_copy_idx:
        j = np.asarray([qb_idx], dtype=int)
        j_prime = np.asarray(list(set(first_copy_idx) - set(j)), dtype=int)
        shifted_j_prime = num_qubits + j_prime

        circuit = QuantumCircuit(
            2 * num_qubits,
            2 * num_qubits - 1,
            name=pdip_circuit_name_prefix + str(qb_idx),
        )
        circuit.compose(pdip_circuit, inplace=True)
        circuit.h(shifted_j_prime)

        if not statevector_mode:
            circuit.measure(shifted_j_prime, shifted_j_prime - 1)
        circuits_to_execute += [circuit]

    return circuits_to_execute


def eval_tests_with_result(result: Result, statevector_mode: bool = False):
    # pylint: disable=too-many-locals
    """
    TODO
    """

    dip_eval = 0.0
    pdip_eval = 0.0

    num_qubits = result.num_qubits // 2
    dip_counts_dict = result.get_counts(0)
    dip_zero_outcome_counts_dict = _get_zero_counts_at_indices(
        dip_counts_dict, np.arange(num_qubits)
    )

    dip_counts_total = sum(list(dip_counts_dict.values()))
    dip_zero_outcome_counts_total = sum(list(dip_zero_outcome_counts_dict.values()))

    dip_eval = dip_zero_outcome_counts_total / dip_counts_total

    if 0 <= dip_eval <= 1.0:
        warnings.warn("Computed probability has a value less than 0.0 or 1.0")

    for j in range(1, num_qubits + 1):
        pdip_counts_dict = result.get_counts(j)
        pdip_zero_outcome_counts_dict = _get_zero_counts_at_indices(
            pdip_counts_dict, [j]
        )

        pdip_counts_total = sum(list(pdip_counts_dict.values()))

        pdip_zero_outcome_keys = list(pdip_zero_outcome_counts_dict.keys())
        pdip_zero_outcome_counts = list(pdip_zero_outcome_counts_dict.values())

        pdip_zero_outcome_counts_total = sum(
            list(pdip_zero_outcome_counts_dict.values())
        )
        pdip_zero_outcome_prob = pdip_zero_outcome_counts_total / pdip_counts_total

        # State vector mode not supported yet.
        if statevector_mode:
            logger.log("State vector mode not supported yet.")

        if 0 <= pdip_zero_outcome_prob <= 1.0:
            warnings.warn("Computed probability has a value less than 0.0 or 1.0")

        for key in pdip_zero_outcome_keys:
            pairs = [
                0 if ii == j else int(key[ii]) and int(key[ii + num_qubits])
                for ii in range(num_qubits)
            ]

            logger.log(" --- Pairs of qubits for PDIP test---")
            logger.log(pairs)

            parity = functools.reduce(lambda x, y: ((-1) ** x) * ((-1) ** y), pairs)

            logger.log(" --- Parirty ---")
            logger.log(parity)

            overlap = parity * pdip_zero_outcome_counts[key]
            pdip_eval += overlap * pdip_zero_outcome_prob / num_qubits

    return pdip_eval, dip_eval


def _get_zero_counts_at_indices(counts_dict: dict, j: Union[list, np.ndarray]):
    """
    TODO
    """

    all_zero_counts_dict = {}
    for key, value in counts_dict.items():
        k = np.fromiter(map(int, key), dtype=int)
        if (k[j] == 0).all():
            all_zero_counts_dict[key] = value

    return all_zero_counts_dict


def get_optimizer_instance(config):
    """Returns optimizer instance based on config."""
    # Addressing some special cases for compatibility with various Qiskit optimizers:
    if config.optimizer_name == "adaptive_SPSA":
        optimizer = SPSA
    else:
        optimizer = getattr(optimizers, config.optimizer_name)
    optimizer_config = {}
    optimizer_arg_names = inspect.signature(optimizer).parameters.keys()
    iter_kw = [kw for kw in ["maxiter", "max_trials"] if kw in optimizer_arg_names][0]
    optimizer_config[iter_kw] = config.maxiter
    if "skip_calibration" in optimizer_arg_names:
        optimizer_config["skip_calibration"] = config.skip_any_optimizer_cal
    if "last_avg" in optimizer_arg_names:
        optimizer_config["last_avg"] = config.spsa_last_average
    if "tol" in optimizer_arg_names:
        optimizer_config["tol"] = config.optimizer_tol
    if "c0" in optimizer_arg_names:
        optimizer_config["c0"] = config.spsa_c0
    if "c1" in optimizer_arg_names:
        optimizer_config["c1"] = config.spsa_c1
    if "learning_rate" in optimizer_arg_names:
        optimizer_config["learning_rate"] = lambda: powerseries(
            config.spsa_c0, 0.602, 0
        )
    if "perturbation" in optimizer_arg_names:
        optimizer_config["perturbation"] = lambda: powerseries(config.spsa_c1, 0.101, 0)

    if config.initial_spsa_iteration_idx:
        if "int_iter" in optimizer_arg_names:
            optimizer_config["int_iter"] = config.initial_spsa_iteration_idx

    if "bootstrap_trials" in optimizer_arg_names:
        optimizer_config["bootstrap_trials"] = config.bootstrap_trials

    logger.log(optimizer_config)
    optimizer_instance = optimizer(**optimizer_config)
    return optimizer_instance
