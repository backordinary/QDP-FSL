# https://github.com/aleksey-uvarov/permutevqe/blob/f9550348c0ed9cbcd27cb2b0cf4e0bbfda0d44ac/vqe_predict.py
"""Here we want to analyze the data for noisy circuits. We want
to understand if there is a classically computable value that would
predict the error in VQE."""

from qiskit import QuantumCircuit
from qiskit.dagcircuit import DAGCircuit, DAGOutNode, DAGOpNode
from qiskit.converters import circuit_to_dag
from qiskit.opflow import X, Z, I, Y, StateFn, PauliOp, OperatorBase, PauliSumOp
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import TwoLocal

import networkx as nx
import warnings
from typing import Union
import json
import numpy as np
from math import factorial
from tqdm import tqdm
from itertools import permutations
import matplotlib.pyplot as plt

from vqe_vs_sum import ising_model, unpack_twolocal, random_pauli_op, permute_circuit


warnings.filterwarnings("ignore", category=DeprecationWarning)


def main():
    experiment_type = "ising_perm"
    experiment_id = "1667903490"
    general_data, permutations_data, ps, qs, h = read_data(experiment_id, experiment_type)
    n_qubits = general_data["n_qubits"]
    # h = ising_model(n_qubits)
    conesums = np.zeros(factorial(n_qubits))
    circ = TwoLocal(n_qubits, ['ry'], 'rxx',
                    entanglement='linear',
                    reps=general_data["depth"])
    circ = unpack_twolocal(circ)

    for i, perm in tqdm(enumerate(permutations(range(n_qubits)))):
        permutation = permutations_data[i, :n_qubits]
        # P = permutatiton_matrix(permutation)
        # Pinv = np.linalg.inv(P)
        # qs_permuted = Pinv @ qs @ P
        # not touching ps for now because that doesn't matter anyway
        h_perm = h.permute(list(perm))
        circ_perm = permute_circuit(circ, perm)
        conesums[i] = cone_weighted_errorsum(h_perm, circ_perm, ps, qs)

    plt.errorbar(conesums,
                 permutations_data[:, -3],
                 permutations_data[:, -2],
                 None, 'o', capsize=5)
    plt.ylabel('E')
    plt.xlabel('Cone-weighted error sum')
    plt.show()


def permutatiton_matrix(perm):
    P = np.zeros((len(perm), len(perm)))
    for i, j in enumerate(perm):
        P[i, int(j)] = 1
    return P


def read_data(experiment_id: str, experiment_type: str):
    """Return arrays of ps, qs, and a json specifying the experiment details"""
    if experiment_type == "ising_perm":
        with open("data/ising_data_"
                  + experiment_id + ".json", "r") as f:
            general_data = json.load(f)
        permutations_data = np.loadtxt('permutations_en_sum_' + experiment_id + '.txt')
        ps = np.loadtxt('data/ps_' + experiment_id + ".txt")
        qs = np.loadtxt('data/qs_' + experiment_id + ".txt")
        h = ising_model(general_data["n_qubits"])
        return general_data, permutations_data, ps, qs, h
    elif experiment_type == "random_paulisumop":
        with open("data/random_paulisumop_data_"
                  + experiment_id + ".json", "r") as f:
            general_data = json.load(f)
        permutations_data = np.loadtxt('permutations_en_sum_' + experiment_id + '.txt')
        ps = np.loadtxt('data/ps_' + experiment_id + ".txt")
        qs = np.loadtxt('data/qs_' + experiment_id + ".txt")
        h = random_pauli_op(general_data["n_qubits"],
                            general_data["hamiltonian_cardinality"],
                            general_data["hamiltonian_seed"])
        return general_data, permutations_data, ps, qs, h
    else:
        raise ValueError("Invalid experiment type")


def cone_weighted_errorsum(h: PauliSumOp,
                           circ: QuantumCircuit,
                           ps: np.array,
                           qs: np.array):
    """For a Hamiltonian \\sum c_k h_k, compute the sum
    \\sum c_k p_j where p_j stands for error rates of gates
    included in the causal cone of the Pauli string h_k"""
    weighted_sum = 0
    for k in h:
        prim = k.primitive
        weighted_sum += k.coeff * cone_sum(prim, circ, ps, qs)
    return weighted_sum


def cone_sum(paulistring: PauliOp,
             circ: QuantumCircuit,
             ps: np.array,
             qs: np.array):
    rate_sum = 0
    dag = circuit_to_dag(circ).to_networkx()
    support = pauli_string_support(paulistring)
    out_nodes = [node for node in dag.nodes
                 if type(node) == DAGOutNode]
    out_node_indices = [node.wire.index for node in out_nodes]
    total_opnodes = set()
    for qubit_index in support:
        node = out_nodes[out_node_indices.index(qubit_index)]
        ancestors = nx.ancestors(dag, node)
        opnodes = {node for node in ancestors
                   if type(node) == DAGOpNode}
        total_opnodes = total_opnodes.union(opnodes)
    for opnode in opnodes:
        gate_qubits = [qubit.index for qubit in opnode.qargs]
        if len(gate_qubits) == 2:
            rate_sum += qs[gate_qubits[0], gate_qubits[1]]
        elif len(gate_qubits) == 1:
            rate_sum += ps[gate_qubits[0]]
    return rate_sum


def pauli_string_support(paulistring: Union[PauliOp, SparsePauliOp]):
    """Given a Pauli string, return a list of qubits in which
    it has nontrivial Pauli matrices"""
    if type(paulistring) == PauliOp:
        ps_crc = paulistring.to_circuit()
        ps_instr = ps_crc.data[0]
        ps_name = ps_instr.operation.params[0][::-1]
    elif type(paulistring) == SparsePauliOp:
        ps_name = paulistring.settings['data'][0][::-1].settings['data']
    else:
        raise TypeError("paulistring has to be a PauliOp or a SparsePauliOp")
    support = [pos for pos, ch in enumerate(ps_name) if ch != "I"]
    return support


if __name__ == "__main__":
    main()

    # experiment_type = "random_paulisumop"
    # experiment_id_1 = "1667217104"
    # experiment_id_2 = "1667242345"
    # experiment_id_3 = "1667277830"
    #
    # data_1 = read_data(experiment_id_1, experiment_type)
    # data_2 = read_data(experiment_id_2, experiment_type)
    # data_3 = read_data(experiment_id_3, experiment_type)
    # # print(data_1)
    # # print(data_1[1].shape)
    # # print(data_1[1][:, -3].shape)
    # # print(data_1[1][:, -1].shape)
    # plt.figure()
    # plt.scatter(data_1[1][:, -1], data_1[1][:, -3])
    # plt.scatter(data_2[1][:, -1], data_2[1][:, -3])
    # plt.scatter(data_3[1][:, -1], data_3[1][:, -3])
    # plt.xlabel('Error sum')
    # plt.ylabel('E')
    # plt.savefig('data/random_paulisumop.png',
    #             format='png',
    #             bbox_inches='tight', dpi=400)
    # plt.show()
    # # circ_test = QuantumCircuit(3)
    # # circ_test.cnot(0, 1)
    # # circ_test.cnot(1, 2)
    # #
    # # ps = np.array([1e-3]) * 3
    # # qs = np.array([[1, 2, 3],
    # #                [4, 5, 6],
    # #                [7, 8, 9]]) * 1e-2
    # #
    # # h = ising_model(3, 1, 1)
    # # print(cone_weighted_errorsum(h, circ_test, ps, qs))

