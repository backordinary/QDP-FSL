# https://github.com/hflash/profiling/blob/0498dff8c1901591d4428c3149eb3aadf80ac483/testfiles/circuit_profiling_debug.py
import os
from qiskit import QuantumCircuit, QuantumRegister
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def main():
    filename = 'fail_examples/adr4_197.qasm'
    circuit = QuantumCircuit.from_qasm_file(filename)
    num_qubits = len(circuit.qubits)
    profiling_array = np.zeros((num_qubits, num_qubits))
    # print(profiling_array)
    #
    # print(len(circuit._data))
    # print(circuit.calibrations)
    gates_list = []
    for i in range(len(circuit._data)):
        item = circuit._data.__getitem__(i)
        if len(item[1]) == 1:
            # print(item)
            # print(item[1][0].index)
            profiling_array[item[1][0].index][item[1][0].index] += 1
            # profiling_array[][]
        if len(item[1]) == 2:
            # print(item[1][0])
            # print(item[1][1])
            # print("cx: ")
            # print(item[1][0].index)
            # print(item[1][1].index)
            profiling_array[item[1][0].index][item[1][1].index] += 1
            gates_list.append([item[1][0].index, item[1][1].index])

    print(profiling_array)
    # print(gates_list)
    graph_gates = nx.Graph()
    graph_gates.add_edges_from(gates_list)
    # # 邻接矩阵
    # A = nx.adjacency_matrix(graph_gates)
    # print('邻接矩阵:\n', A.todense())
    #
    # # 关联矩阵
    # I = nx.incidence_matrix(graph_gates)
    # print('\n关联矩阵:\n', I.todense())
    #
    # # 拉普拉斯矩阵
    # L = nx.laplacian_matrix(graph_gates)
    # print('\n拉普拉斯矩阵:\n', L.todense())
    #
    # # 标准化的拉普拉斯矩阵
    # NL = nx.normalized_laplacian_matrix(graph_gates)
    # print('\n标准化的拉普拉斯矩阵:\n', NL.todense())
    #
    # # 有向图拉普拉斯矩阵
    # DL = nx.directed_laplacian_matrix(graph_gates)
    # print('\n有向拉普拉斯矩阵:\n', DL)
    #
    # # 拉普拉斯算子的特征值
    # LS = nx.laplacian_spectrum(graph_gates)
    # print('\n拉普拉斯算子的特征值:\n', LS)
    #
    # # 邻接矩阵的特征值
    # AS = nx.adjacency_spectrum(graph_gates)
    # print('\n邻接矩阵的特征值:\n', AS)
    #
    # # 无向图的代数连通性
    # AC = nx.algebraic_connectivity(graph_gates)
    # print('\n无向图的代数连通性:\n', AC)
    nx.draw(graph_gates)
    plt.show()
    # a = lap.normalized_laplacian(profiling_array)
    # print(a)
    # b = np.linalg.eigvals(a)
    # print(b)
    connect = nx.algebraic_connectivity(graph_gates, normalized=True, method='lanczos')
    # lap = nx.normalized_laplacian_matrix(graph_gates)
    # print(lap.todense())
    # a = np.linalg.eigh(lap.todense())
    # print(a)
    # print(b)
    print(connect)
    # print(nx.edge_connectivity())
    plt.matshow(profiling_array, cmap=plt.cm.Blues)
    plt.show()


if __name__ == '__main__':
    main()
