# https://github.com/hflash/profiling/blob/0498dff8c1901591d4428c3149eb3aadf80ac483/chip_profiling.py
import networkx as nx
from qiskit import IBMQ
# from qiskit.transpiler import CouplingMap
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=3)

def main():
    IBMQ.load_account()
    provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
    """valencia: 5, santiago: 5, melbourne: 15, yorktown - ibmqx2"""
    D = ['ibmqx2', 'ibmq_valencia', 'ibmq_santiago', 'ibmq_16_melbourne']
    # D1 = ['ibmqx2', 'ibmq_burlington', 'ibmq_essex', 'ibmq_london', 'ibmq_ourense', 'ibmq_vigo', 'ibmq_16_melbourne']

    N = {'ibmqx2': 5, 'ibmq_valencia': 5, 'ibmq_santiago': 5, 'ibmq_16_melbourne': 15}
    chip_ibmqx2 = [[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1], [2, 3], [2, 4], [3, 2], [3, 4], [4, 2], [4, 3]]
    chip_valencia = [[0, 1], [1, 0], [1, 2], [1, 3], [2, 1], [3, 1], [3, 4], [4, 3]]
    chip_santiago = [[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2], [3, 4], [4, 3]]
    chip_melbourne = [[0, 1], [0, 14], [1, 0], [1, 2], [1, 13], [2, 1], [2, 3], [2, 12], [3, 2], [3, 4], [3, 11],
                      [4, 3], [4, 5], [4, 10], [5, 4], [5, 6], [5, 9], [6, 5], [6, 8], [7, 8], [8, 6], [8, 7], [8, 9],
                      [9, 5], [9, 8], [9, 10], [10, 4], [10, 9], [10, 11], [11, 3], [11, 10], [11, 12], [12, 2],
                      [12, 11], [12, 13], [13, 1], [13, 12], [13, 14], [14, 0], [14, 13]]
    chip_array_ibmqx2 = np.zeros((5, 5))
    chip_array_valencia = np.zeros((5, 5))
    chip_array_santiago = np.zeros((5, 5))
    chip_array_melbourne = np.zeros((15, 15))
    for item in chip_ibmqx2:
        chip_array_ibmqx2[item[0]][item[1]] = 1
    for item in chip_valencia:
        chip_array_valencia[item[0]][item[1]] = 1
    for item in chip_santiago:
        chip_array_santiago[item[0]][item[1]] = 1
    for item in chip_melbourne:
        chip_array_melbourne[item[0]][item[1]] = 1
    graph_ibmx2 = nx.Graph()
    graph_ibmx2.add_nodes_from([0, 1, 2, 3, 4])
    graph_ibmx2.add_edges_from(chip_ibmqx2)
    nx.draw(graph_ibmx2)
    plt.show()
    print(nx.algebraic_connectivity(graph_ibmx2, normalized=True, method='lanczos'))

    graph_valencia = nx.Graph()
    graph_valencia.add_nodes_from([0, 1, 2, 3, 4])
    graph_valencia.add_edges_from(chip_valencia)
    nx.draw(graph_valencia)
    plt.show()
    print(nx.algebraic_connectivity(graph_valencia, normalized=True, method='lanczos'))

    graph_santiago = nx.Graph()
    graph_santiago.add_nodes_from([0, 1, 2, 3, 4])
    graph_santiago.add_edges_from(chip_santiago)
    nx.draw(graph_santiago)
    plt.show()
    print(np.array(nx.adjacency_matrix(graph_santiago).todense()))
    print(nx.algebraic_connectivity(graph_santiago, normalized=True, method='lanczos'))

    graph_melbourne = nx.Graph()
    graph_melbourne.add_nodes_from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    graph_melbourne.add_edges_from(chip_melbourne)
    nx.draw(graph_melbourne)
    plt.show()
    print(np.array(nx.adjacency_matrix(graph_melbourne).todense()))
    connect = nx.algebraic_connectivity(graph_melbourne, normalized=True, method='lanczos')
    print(connect)
    # print(chip_array_ibmqx2)
    # for d in D:
    #     backend = provider.get_backend(d)
    #     print(backend.configuration().coupling_map)
    # backend = provider.get_backend('ibmq_16_melbourne')
    # print(backend.configuration().coupling_map)
    # print(type(backend.configuration().coupling_map))


if __name__ == '__main__':
    main()
