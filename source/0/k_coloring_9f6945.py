# https://github.com/kevinab107/q-algos/blob/56cb8471e2b334af8f94256b51eba9fa128ea769/k_coloring.py
#%%
from qiskit.aqua.algorithms.adaptive import QAOA
from qiskit import BasicAer

from qiskit.aqua.algorithms import VQE
from qiskit.quantum_info import Pauli, Operator
from qiskit.aqua.operators import WeightedPauliOperator
from q_utils import get_test_graph
from qiskit.aqua.components.variational_forms import RYRZ
from qiskit.aqua.components.optimizers import L_BFGS_B, COBYLA, NELDER_MEAD
from qiskit.aqua import QuantumInstance
import numpy as np
import networkx as nx
from q_utils import get_compute_backend


def get_operator(graph, k):

    weight_matrix = nx.to_numpy_array(graph)
    num_nodes = graph.number_of_nodes()
    edges = graph.number_of_edges()
    num_qubits = num_nodes * k
    operator_list = []
    z_p = np.zeros(num_qubits, dtype=np.bool)
    x_p = np.zeros(num_qubits, dtype=np.bool)
    weight_offset = 0
    for edge in graph.edges():
        for color in range(k):
            if weight_matrix[edge] != 0:
                z_p[edge[0] * k + color] = True
                z_p[edge[1] * k + color] = True
                operator_list.append([0.25, Pauli(z_p, x_p)])
                z_p[edge[0] * k + color] = False
                z_p[edge[1] * k + color] = False

                z_p[edge[0] * k + color] = True
                operator_list.append([-0.25, Pauli(z_p, x_p)])
                z_p[edge[0] * k + color] = False

                z_p[edge[1] * k + color] = True
                operator_list.append([-0.25, Pauli(z_p, x_p)])
                z_p[edge[1] * k + color] = False

                # weight_offset -= 0.5 * weight_matrix[edge]
    H_c = WeightedPauliOperator(paulis=operator_list)
    return H_c, weight_offset


def get_mixing_operator(graph, k):
    num_nodes = graph.number_of_nodes()
    num_qubits = num_nodes * k
    z_p = np.zeros(num_qubits, dtype=np.bool)
    x_p = np.zeros(num_qubits, dtype=np.bool)
    operator_list = []
    for node in range(num_nodes):
        for color in range(k):
            color_next = (color + 1) % k
            x_p[node * k + color] = True
            x_p[node * k + color_next] = True
            operator_list.append([1 / k, Pauli(z_p, x_p)])
            x_p[node * k + color_next] = False
            x_p[node * k + color] = False

    H_m = WeightedPauliOperator(paulis=operator_list)
    return H_m


def solve_kcoloring_qaoa(graph, solve_type, k):
    backend = get_compute_backend(solve_type)
    operator, offset = get_operator(graph, k)
    mixing_operator = get_mixing_operator(graph, k)
    qaoa = QAOA(operator, p=1, optimizer=NELDER_MEAD(maxfev=1000))
    result = qaoa.run(QuantumInstance(backend))
    return result
