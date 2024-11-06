# https://github.com/Andris-Huang/Quantum-Course-Scheduler/blob/6809294a81587b9cbd6eba6634b4e97edc59187e/qaoa.py
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import numpy as np
import nlopt
import networkx as nx
import qiskit
from qiskit import Aer
from qiskit.optimization.applications.ising import max_cut
from qiskit.optimization.applications.ising.common import sample_most_likely
from qiskit.aqua import QuantumInstance
from qiskit.aqua.components.optimizers.nlopts.esch import ESCH
from qiskit.aqua.algorithms import QAOA
from qiskit.quantum_info import Pauli
from qiskit.aqua.operators.legacy.weighted_pauli_operator import WeightedPauliOperator
import pandas as pd

import os
import itertools
import random
import dataset
import utils
import time

csv = dataset.CSV()

def max_cut_solver(graph, output_dir, p_steps=1, save_fig=False, print_result=False):
    """
    Perform the max-cut solver by dwave and return the graph size and solving time.
    Input:
        graph: graph input
        output_dir: the output directory
        p_step: the depth for QAOA circuits
        save_fig: save the figure iff true
        print_result: boolean for display grouping result
    Return:
        result: [S0, S1], node index for two groups
        log: [n_node, delta_t], a list with number of nodes and time took to solve max-cut
    """

    n_nodes = graph["n_node"]
    edges = graph["edges"]
    edge_labels = graph["edge_labels"]

    plot_fig = n_nodes <= 10
    save_fig = save_fig and plot_fig

    # ------- Set up our graph -------

    # Create empty graph
    G = nx.Graph()

    # Add attritubes to the graph
    G.add_weighted_edges_from(edges)

    # Save the graph beforehand for testing/demo purposes
    if save_fig:
        g = G.copy()
        plt.figure()
        pos = nx.spring_layout(G)
        nx.draw_networkx(g, pos, node_color='r')
        nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels)
        filename = "Input Graph.png"
        out_name = os.path.join(output_dir, filename)
        plt.savefig(out_name, bbox_inches='tight')

    
    # Weighted matrix
    n = n_nodes
    w = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            temp = G.get_edge_data(i, j, default=0)
            if temp != 0:
                w[i, j] = temp['weight']

    print(">>> Graph created successfully, start solving max-cut")
    num_nodes = w.shape[0]
    pauli_list = []
    for i in range(num_nodes):
        for j in range(i):
            if w[i, j] != 0:
                xp = np.zeros(num_nodes, dtype=np.bool)
                zp = np.zeros(num_nodes, dtype=np.bool)
                zp[i] = True
                zp[j] = True
                pauli_list.append([0.5 * w[i, j], Pauli(zp, xp)])
    qubitOp = WeightedPauliOperator(paulis=pauli_list)

    n_shots = 512

    backend = Aer.get_backend('aer_simulator')
    quantum_instance = QuantumInstance(backend, shots=n_shots)

    qaoa = QAOA(qubitOp, ESCH(max_evals=50), p=p_steps+8)
    result = qaoa.run(quantum_instance)

    solution = sample_most_likely(result['eigvecs'][0])
    dt = result['eval_time']
    delta_t = utils.time_lasted(dt)
    print(f">>> Grouping finished, total time: {delta_t}")

    S0 = [node for node in range(n_nodes) if solution[node]==0]
    S1 = [node for node in range(n_nodes) if solution[node]==1]
    uncut_edges = list(itertools.combinations(S0, 2)) + list(itertools.combinations(S1, 2))
    cut_edges = [i for i in edge_labels if i not in uncut_edges]

    # Display best result
    if save_fig:
        plt.figure()
        nx.draw_networkx_nodes(G, pos, nodelist=S0, node_color='r')
        nx.draw_networkx_nodes(G, pos, nodelist=S1, node_color='c')
        nx.draw_networkx_edges(G, pos, edgelist=cut_edges, style='dashdot', alpha=0.5, width=3)
        nx.draw_networkx_edges(G, pos, edgelist=uncut_edges, style='solid', width=3)
        nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels)
        nx.draw_networkx_labels(G, pos)

        filename = "Output Graph.png"
        out_name = os.path.join(output_dir, filename)
        plt.savefig(out_name, bbox_inches='tight')
        print(">>> Your plot is saved to {}".format(out_name))

    log = [n_nodes, dt]
    result_group = [S0, S1]
    
    return result_group, log