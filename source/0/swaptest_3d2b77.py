# https://github.com/nahumsa/volta/blob/77c060b56162ca7eb0072a594f5470d75ccb81ed/volta/swaptest.py
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


import qiskit
from qiskit import QuantumCircuit, execute
from qiskit.utils import QuantumInstance
from qiskit.providers import BaseBackend

from typing import Union
import textwrap


def swap_test_circuit(qc1: QuantumCircuit, qc2: QuantumCircuit) -> QuantumCircuit:
    """Construct the SWAP test circuit given two circuits.

    Args:
        qc1(qiskit.QuantumCircuit): Quantum circuit for the
        first state.
        qc2(qiskit.QuantumCircuit): Quantum circuit for the
        second state.
    Output:
        (qiskit.QuantumCircuit): swap test circuit.
    """
    # Helper variables
    n_total = qc1.num_qubits + qc2.num_qubits
    range_qc1 = [i + 1 for i in range(qc1.num_qubits)]
    range_qc2 = [i + qc1.num_qubits + 1 for i in range(qc2.num_qubits)]

    # Constructing the SWAP test circuit
    qc_swap = QuantumCircuit(n_total + 1, 1)
    qc_swap.append(qc1, range_qc1)
    qc_swap.append(qc2, range_qc2)

    # Swap Test
    qc_swap.h(0)
    for index, qubit in enumerate(range_qc1):
        qc_swap.cswap(0, qubit, range_qc2[index])
    qc_swap.h(0)

    # Measurement on the auxiliary qubit
    qc_swap.measure(0, 0)
    return qc_swap


def measure_swap_test(
    qc1: QuantumCircuit,
    qc2: QuantumCircuit,
    backend: Union[BaseBackend, QuantumInstance],
    num_shots: int = 10000,
) -> float:
    """Returns the fidelity from a SWAP test.

    Args:
        qc1 (QuantumCircuit): Quantum Circuit for the first state.
        qc2 (QuantumCircuit): Quantum Circuit for the second state.
        backend (Union[BaseBackend,QuantumInstance]): Backend.
        num_shots (int, optional): Number of shots. Defaults to 10000.

    Returns:
        float: result of the overlap betweeen the first and second state.
    """
    swap_circuit = swap_test_circuit(qc1, qc2)

    # Check if the backend is a quantum instance.
    if qiskit.utils.quantum_instance.QuantumInstance == type(backend):
        count = backend.execute(swap_circuit).get_counts()
    else:
        count = (
            execute(swap_circuit, backend=backend, shots=num_shots)
            .result()
            .get_counts()
        )

    if "0" not in count:
        count["0"] = 0
    if "1" not in count:
        count["1"] = 0

    total_counts = count["0"] + count["1"]
    fid_meas = count["0"]
    p_0 = fid_meas / total_counts
    return 2 * (p_0 - 1 / 2)


def dswap_test_circuit(qc1: QuantumCircuit, qc2: QuantumCircuit) -> QuantumCircuit:
    """Construct the destructive SWAP test circuit given two circuits.

    Args:
        qc1(qiskit.QuantumCircuit): Quantum circuit for the
        first state.
        qc2(qiskit.QuantumCircuit): Quantum circuit for the
        second state.
    Output:
        (qiskit.QuantumCircuit): swap test circuit.
    """
    # Helper variables
    n_total = qc1.num_qubits + qc2.num_qubits
    range_qc1 = [i for i in range(qc1.num_qubits)]
    range_qc2 = [i + qc1.num_qubits for i in range(qc2.num_qubits)]

    # Constructing the SWAP test circuit
    qc_swap = QuantumCircuit(n_total, n_total)
    qc_swap.append(qc1, range_qc1)
    qc_swap.append(qc2, range_qc2)

    for index, qubit in enumerate(range_qc1):
        qc_swap.cx(qubit, range_qc2[index])
        qc_swap.h(qubit)

    for index, qubit in enumerate(range_qc1):
        qc_swap.measure(qubit, 2 * index)

    for index, qubit in enumerate(range_qc2):
        qc_swap.measure(range_qc2[index], 2 * index + 1)
    return qc_swap


def measure_dswap_test(
    qc1: QuantumCircuit,
    qc2: QuantumCircuit,
    backend: Union[BaseBackend, QuantumInstance],
    num_shots: int = 10000,
) -> float:
    """Returns the fidelity from a destructive SWAP test.

    Args:
        qc1 (QuantumCircuit): Quantum Circuit for the first state.
        qc2 (QuantumCircuit): Quantum Circuit for the second state.
        backend (Union[BaseBackend,QuantumInstance]): Backend.
        num_shots (int, optional): Number of shots. Defaults to 10000.

    Returns:
        float: result of the overlap betweeen the first and second state.
    """
    n = qc1.num_qubits
    swap_circuit = dswap_test_circuit(qc1, qc2)

    # Check if the backend is a quantum instance.
    if qiskit.utils.quantum_instance.QuantumInstance == type(backend):
        count = backend.execute(swap_circuit).get_counts()
    else:
        count = (
            execute(swap_circuit, backend=backend, shots=num_shots)
            .result()
            .get_counts()
        )

    result = 0
    for meas, counts in count.items():
        split_meas = textwrap.wrap(meas, 2)
        for m in split_meas:
            if m == "11":
                result -= counts
            else:
                result += counts

    total = sum(count.values())

    return result / (n * total)


def amplitude_transition_test_circuit(
    qc1: QuantumCircuit, qc2: QuantumCircuit
) -> QuantumCircuit:
    """Construct the SWAP test circuit given two circuits using amplitude transition.
    Args:
        qc1(qiskit.QuantumCircuit): Quantum circuit for the
        first state.
        qc2(qiskit.QuantumCircuit): Quantum circuit for the
        second state.
    Output:
        (qiskit.QuantumCircuit): swap test circuit.
    """

    # Constructing the SWAP test circuit
    n_qubits = qc1.num_qubits
    qc_swap = QuantumCircuit(n_qubits)
    qc_swap.append(qc1, range(n_qubits))
    qc_swap.append(qc2.inverse(), range(n_qubits))

    # Measurement on the auxiliary qubit
    qc_swap.measure_all()
    return qc_swap


def measure_amplitude_transition_test(
    qc1: QuantumCircuit,
    qc2: QuantumCircuit,
    backend,
    num_shots: int = 10000,
) -> float:
    """Returns the fidelity from a SWAP test using amplitude transition.
    Args:
        qc1 (QuantumCircuit): Quantum Circuit for the first state.
        qc2 (QuantumCircuit): Quantum Circuit for the second state.
        backend (Union[BaseBackend,QuantumInstance]): Backend.
        num_shots (int, optional): Number of shots. Defaults to 10000.
    Returns:
        float: result of the overlap betweeen the first and second state.
    """
    swap_circuit = amplitude_transition_test_circuit(qc1, qc2)

    # Check if the backend is a quantum instance.
    if qiskit.utils.quantum_instance.QuantumInstance == type(backend):
        count = backend.execute(swap_circuit).get_counts()
    else:
        count = (
            execute(swap_circuit, backend=backend, shots=num_shots)
            .result()
            .get_counts()
        )

    if "0" * qc1.num_qubits not in count:
        count["0" * qc1.num_qubits] = 0

    total_counts = sum(count.values())
    fid_meas = count["0" * qc1.num_qubits]
    p_0 = fid_meas / total_counts
    return p_0
