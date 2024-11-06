# https://github.com/UST-QuAntiL/quantum-circuit-generator/blob/c3c41df4441043f575c61817922a602b38d5c650/app/services/algorithms/vqe_algorithm.py
import numpy as np

from qiskit import QuantumRegister
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.algorithms import VQE


class VQEAlgorithm:
    @classmethod
    def create_circuit(cls, ansatz, parameters, observable):
        """
        :param ansatz: QuantumCircuit (from qasm string) instance describing the ansatz.
                       If None (no ansatz) is given using RealAmplitudes ansatz.
        :param parameters: Parameters for the ansatz circuit
        :param observable: Qubit operator of the Observable given as pauli string
        :return: OpenQASM Circuit of the VQE ansatz

        Returns a circuit that consists of ansatz and preparation for measurement with observable.
        If no custom ansatz is given, RealAmplitudes ansatz is chosen.
        If a custom ansatz is given, no parameters shall be provided.
        """

        vqe = VQE(ansatz=ansatz)
        vqe_qc = vqe.construct_circuit(parameters, observable)[0]

        # decompose default ansatz
        if ansatz is None:
            vqe_qc = vqe_qc.decompose("RealAmplitudes")

        return vqe_qc
