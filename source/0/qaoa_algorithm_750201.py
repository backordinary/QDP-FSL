# https://github.com/mbeisel/VQA-workflow/blob/6c7ec57dd4826f2130b6d72735c118ca35f04ffb/services/circuit-generation-service/api/services/algorithms/qaoa_algorithm.py
import numpy as np

from qiskit import QuantumRegister
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.algorithms.linear_solvers.hhl import HHL

from api.services.encodings.amplitude_encoding import AmplitudeEncoding

from qiskit.algorithms import QAOA
from qiskit.algorithms.optimizers import COBYLA
from qiskit.quantum_info import Pauli
from qiskit.opflow import PauliSumOp


class QAOAAlgorithm:
    @classmethod
    def create_operator(cls, weight_matrix):
        """
        :param matrix: adjacency matrix describing the undirected graph

        Create Hamiltonian operator to be used in QAOA.
        Here we transform an adjacency matrix into an operator to be used in the QAOA circuit.
        """
        weight_matrix = np.array(weight_matrix)
        num_nodes = weight_matrix.shape[0]
        pauli_list = []

        for i in range(num_nodes):
            for j in range(i):
                if weight_matrix[i, j] != 0:
                    x_p = np.zeros(num_nodes, dtype=bool)
                    z_p = np.zeros(num_nodes, dtype=bool)
                    z_p[i] = True
                    z_p[j] = True
                    pauli_list.append([0.5, Pauli((z_p, x_p))])

        pauli_list = [(pauli[1].to_label(), pauli[0]) for pauli in pauli_list]

        return PauliSumOp.from_list(pauli_list)

    @classmethod
    def create_circuit(cls, adj_matrix, beta, gamma):
        """
        :param adj_matrix: adjacency matrix describing the undirected graph
        :param beta: beta parameter used in qaoa
        :param gamma: gamma parameter used in qaoa
        :return: OpenQASM Circuit

        Creates circuit used in qaoa for MaxCut of undirected graph.
        Custom AmplitudeEncoding is used for vector preparation.
        """

        operator = cls.create_operator(adj_matrix)
        optimizer = COBYLA()
        qaoa = QAOA(optimizer)
        qaoa_qc = qaoa.construct_circuit([gamma, beta], operator)[0]

        return qaoa_qc
