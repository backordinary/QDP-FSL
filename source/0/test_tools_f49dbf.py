# https://github.com/FilipeChagasDev/register-by-constant-qft-adder/blob/f05c6998f00b5f06b4119ffced5dc4f31c0b9113/test_tools.py
# By Filipe Chagas
# 2022

import qiskit
from numeric_systems import *
from typing import *

def result_string_to_list(result: str) -> List[bool]:
    """Convert a string result to a list of bits.

    :param result: String result from Qiskit.
    :type result: str
    :return: List of bits. Most significant bit last.
    :rtype: List[bool]
    """
    return [(True if c == '1' else False) for c in result[::-1]]

def extract_register_from_result(result: List[bool], register: List[int]) -> List[bool]:
    """Extract a register from a list of bits.

    :param result: List of bits from Qiskit result. Most significant bit last.
    :type result: List[bool]
    :param register: List of register's qubits indexes. Most significant bit last.
    :type register: List[int]
    :return: List of bits. Most significant bit last.
    :rtype: List[bool]
    """
    return [result[i] for i in register]

def one_shot_simulation(circuit: qiskit.QuantumCircuit) -> List[bool]:
    """Do a one-shot simulation with Qiskit and return it's result as a list of bits.

    :param circuit: Qiskit quantum circuit.
    :type circuit: qiskit.QuantumCircuit
    :return: List of bits. Most significant bit last.
    :rtype: List[bool]
    """
    backend = qiskit.BasicAer.get_backend('qasm_simulator')
    job = qiskit.execute(circuit, backend, shots=1)
    counts = job.result().get_counts()
    result_string = list(counts.keys())[0]
    return result_string_to_list(result_string)
