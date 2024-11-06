# https://github.com/mgrzesiuk/qiskit_utils/blob/3666b3ea2074ce6ee75fa22a8165880ca147c71d/qiskit_utils/parse_counts.py
from typing import Dict, Union, List, Set

from qiskit import QuantumCircuit
from qiskit.circuit import Clbit, Measure
from qiskit.result import Result


def parse_counts(qiskit_result: Result, circuit: QuantumCircuit, measurement_names: Set[str]={Measure().name},) -> Dict[str, int]:
    """
    parse results into a dictionary similar to what results.get_counts() returns but accessing measurement of qubit with index i is done via key[i]
    where key is key of results.get_counts() (that is a bitstring showing state)
    :param qiskit_result: result returned by backend.run(circuit)
    :param circuit: circuit for which the qiskit_result was run
    :return: dictionary containing parsed counts
    """
    qubit_clbit_mapping = _get_qubit_mapping(circuit, measurement_names)
    parsed_results = {}
    for state, count in qiskit_result.get_counts().items():
        parsed_state = _parse_state(state)
        new_state = ['-']*len(circuit.qubits)
        for qubit_index in range(len(circuit.qubits)):
            measurement_index = qubit_clbit_mapping[qubit_index]

            if measurement_index is None:
                continue

            new_state[qubit_index] = parsed_state[measurement_index]
        parsed_count = ''.join(new_state)
        if parsed_count in parsed_results:
            parsed_results[parsed_count] += count
        else:
            parsed_results[parsed_count] = count


    return parsed_results


def _get_qubit_mapping(circuit: QuantumCircuit, measurement_names: Set[str]) -> List[Union[None, int]]:
    """
    return mapping between index of a qubit and a real index of its
    measurement result in the state returned by get_counts(). indices do not consider spaces so state must have spaces
    removed before indices address correctly
    :param circuit: circuit for which to create the mapping
    :return: mapping between qubit indices and its measurement indices
    """
    qubit_mapping = [None] * circuit.num_qubits
    for instruction, qubits, bits in circuit.data:
        if instruction.name in measurement_names:
            qubit_index = circuit.qubits.index(qubits[0])
            qubit_mapping[qubit_index] = _get_real_clbit_index(circuit, bits[0])
    return qubit_mapping


def _get_real_clbit_index(circuit: QuantumCircuit, clbit: Clbit) -> int:
    """
    return the index of classical bit in the state returned by counts (indices don't consider spaces
    - they must be removed for the indices to address correctly)
    :param circuit: circuit that clbit is part of
    :param clbit: clbit to get index of
    :return:
    """
    creg_index = 0
    for creg in reversed(circuit.cregs):
        if clbit in creg:
            clbit_index = creg.index(clbit)
            return creg_index + (len(creg) - clbit_index - 1)
        else:
            creg_index += len(creg)


def _parse_state(state: str) -> str:
    """
    removes spaces from str
    :param state: str to parse
    :return: provided str without spaces
    """
    return state.replace(" ", "")
