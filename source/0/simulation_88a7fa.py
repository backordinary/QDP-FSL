# https://github.com/rhlst/qaoa-for-the-knapsack-problem/blob/2a67fac48ee370392554b2bd349c4b2d9f4e1dde/code/simulation.py
"""Definitions and helper functions for circuit simulation using qiskit."""
from qiskit import Aer


backend = Aer.get_backend("aer_simulator_statevector")


def get_statevector(transpiled_circuit, parameter_dict):
    bound_circuit = transpiled_circuit.bind_parameters(parameter_dict)
    result = backend.run(bound_circuit, shots=1).result()
    statevector = result.get_statevector()
    return statevector
