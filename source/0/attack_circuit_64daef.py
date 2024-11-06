# https://github.com/OrenScheer/certified-deletion/blob/d5e4f8b65bbdcde105dcc5f60ec1901b5b9e2118/attack_circuit.py
"""Circuits used by the receiving party to maliciously attempt to break the certified deletion."""

from typing import List
from qiskit import QuantumCircuit
from states import Ciphertext
from math import cos, sin, pi


def breidbart_measure(ciphertext: Ciphertext) -> List[QuantumCircuit]:
    """Creates and returns a circuit with a Breidbart measurement, given a ciphertext."""
    breidbart_matrix = [
        [cos(pi/8), sin(pi/8)],
        [-sin(pi/8), cos(pi/8)]
    ]
    attack_circuits = [circuit.copy() for circuit in ciphertext.circuits]
    for circuit in attack_circuits:
        circuit.barrier()
        circuit.unitary(                                 # type: ignore
            breidbart_matrix, range(circuit.num_qubits), label='breidbart')
        circuit.measure_all()
    return attack_circuits
