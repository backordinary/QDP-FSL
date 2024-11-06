# https://github.com/nahumsa/volta/blob/10e927569292fcae6371115d53e7fdf3d006cd15/volta/ansatz.py
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


from qiskit.circuit.library import TwoLocal
from qiskit import QuantumCircuit
import numpy as np


def _get_ansatz(n_qubits: int, reps: int = 1) -> QuantumCircuit:
    """Create a TwoLocal ansatz for `n_qubits`.

    Args:
        n_qubits (int, optional): number of qubits for the ansatz

    Returns:
        QuantumCircuit: TwoLocal circuit. defaults to 1.
    """
    return TwoLocal(n_qubits, ["ry", "rz"], "cx", reps=reps)


def get_num_parameters(n_qubits: int) -> int:
    """Get the number of parameters for a TwoLocal
    Ansatz.

    Args:
        n_qubits (int): number of qubits

    Returns:
        int: number of parameters
    """
    return len(_get_ansatz(n_qubits).parameters)


def get_var_form(params: np.array, n_qubits: int = 2, reps: int = 1) -> QuantumCircuit:
    """Get an hardware-efficient ansatz for n_qubits
    given parameters.

    Args:
        params (np.array): parameters to be applied on the circuit.
        n_qubits (int, optional): number of qubits. Defaults to 2.
        reps (int, optional): repetition for the ansatz. Defaults to 1.

    Returns:
        QuantumCircuit: Circuit with parameters applied
    """

    # Define variational Form
    var_form = _get_ansatz(n_qubits, reps)

    # Get Parameters from the variational form
    var_form_params = sorted(var_form.parameters, key=lambda p: p.name)

    # Check if the number of parameters is compatible
    assert len(var_form_params) == len(params), "The number of parameters don't match"

    # Create a dictionary with the parameters and values
    param_dict = dict(zip(var_form_params, params))

    # Assing those values for the ansatz
    wave_function = var_form.assign_parameters(param_dict)

    return wave_function
