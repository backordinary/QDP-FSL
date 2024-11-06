# https://github.com/hflash/profiling/blob/0498dff8c1901591d4428c3149eb3aadf80ac483/circuit_profiling.py
import os
from qiskit import QuantumCircuit, QuantumRegister
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import json
from matplotlib.backends.backend_pdf import PdfPages


def main():
    output_list = []
    # filename = './examples/test.qasm'
    # filename = './examples/rd73_252.qasm'
    for filename in os.listdir('./example1'):
    # for filename in os.listdir('./test_cases/qft'):
        # if filename.endswith('.qasm'):
        print(filename)
        fname = filename
        filename = './example1/' + filename
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

        graph_gates = nx.Graph()
        graph_gates.add_edges_from(gates_list)
        connect = nx.algebraic_connectivity(graph_gates, normalized=True, method='lanczos')
        output_dic = {}
        output_dic['name'] = fname
        output_dic['matrix'] = profiling_array.tolist()
        # output_dic['connectivity'] = connect
        output_list.append(output_dic)
        s = json.dumps(output_dic)
        f = open("matrix_v1_simple.json", "a+")
        f.write(s)
        f.close()
        # print(profiling_array)
        # print(gates_list)

        # print(np.array(nx.adjacency_matrix(graph_gates).todense()))
        # nx.draw(graph_gates)
        # plt.show()
        # f = open("testfiles/matrix.txt", "a")
        # list_profiling = profiling_array.tolist()
        # c = json.dumps(list_profiling)
        # f.write(filename + '\n')
        # f.write(c + '\n')
        plt.matshow(profiling_array, cmap=plt.cm.Blues)
        plt.show()
        # plt.imshow(profiling_array)

        # lap = nx.laplacian_matrix(graph_gates)
        # a, b = np.linalg.eig(lap)
        # print(a)
        # print(b)
        # connect = nx.edge_connectivity(graph_gates)
        # print(connect)

    # with open("matrix.txt", "a") as f:
    #     list_profiling = profiling_array.tolist()
    #     c = json.dumps(list_profiling)
    #     f.write(filename + '\n')
    #     f.write(c + '\n')
    # s = json.dumps(output_list)
    # f = open("matrix_v1_simple.json", "w")
    # f.write(s)
    # f.close()

    ff = open("matrix_v1_simple.json", "r")
    ss = ff.read()

    content = json.loads(ss)

    for item in content:
        n = item["name"]
        arr = item["matrix"]
        arr_np = np.array(arr)

        print(n)
        print(arr_np)


if __name__ == '__main__':
    main()
