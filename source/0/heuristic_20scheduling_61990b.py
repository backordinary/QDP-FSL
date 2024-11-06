# https://github.com/hflash/profiling/blob/0498dff8c1901591d4428c3149eb3aadf80ac483/heuristic%20scheduling.py
import time
from qiskit import dagcircuit, transpile, QuantumRegister
from qiskit import (
    QuantumCircuit,
    execute,
    Aer,
    converters,
    IBMQ)
from qiskit.circuit import Qubit
import copy
import random
from qiskit.transpiler import CouplingMap, Layout
import matplotlib.pyplot as plt
from qiskit import transpiler


def heuristic_scheduling(qr, front_layer, coupling_graph, coupling_map_list, initial_layout, dag_circuit):
    layout = initial_layout
    swap_schedule = []
    execute_schedule = []
    layers = 3
    # foresee layers
    # node state:
    # 0: could not be executed now
    # 1: in front layer
    # 2: in execute_gate_list
    # 3: the dependency executed
    node_state_dict = {}
    for node in dag_circuit.gate_nodes():
        node_state_dict[node._node_id] = 0
    # print("node_state:", node_state_dict)
    while front_layer:
        execute_gate_list = []
        successor_gate_list = []
        predecessor_gate_list = []
        # check the executionability of the gate in front layer
        for node in front_layer:
            # if node.name == "barrier"
            #     dag_circuit.
            node_state_dict[node._node_id] = 1
            print(node.qargs, node._node_id)
            gate = [node.qargs[0].index, node.qargs[1].index]
            if gate_executable(qr, gate, coupling_map_list, layout):
                execute_gate_list.append(node)
                execute_schedule.append([node.qargs[0].index, node.qargs[1].index])
                # print("node.qargs:", node.qargs)
                node_state_dict[node._node_id] = 2
            # print("execute_gate_list", execute_gate_list)
            # print("front_layer_1", front_layer)

        if execute_gate_list:
            for node in execute_gate_list:
                front_layer.remove(node)
                successor_gate_list = dag_circuit.quantum_successors(node)
                # print()
                for node in successor_gate_list:
                    # if len(node.qargs) == 2:
                    if node.name == 'cx':
                        # predecessor_gate_list = dag_circuit.quantum_predecessors(node)
                        for pre in dag_circuit.predecessors(node):
                            if pre.name == 'cx':
                                if node_state_dict[pre._node_id] == 2:
                                    node_state_dict[node._node_id] = 3
                                else:
                                    node_state_dict[node._node_id] = 0
                                    break
                        # check
                        # done
                        # 存在重复添加的问题
                        # done
                        # 对于node要做识别，哪些是实际操作的node
                        if node_state_dict[node._node_id] == 3:
                            front_layer.append(node)
                            node_state_dict[node._node_id] = 1
            temp_set_frontlayer = set(front_layer)
            front_layer = list(temp_set_frontlayer)
            # print("successor_gate_list", successor_gate_list)
            # print("front_layer_2", front_layer)

            # print(node_state_dict)
        else:
            swap_candidate = swap_candidate_tuple(layout, coupling_map_list)
            score = [0] * len(swap_candidate)
            swap_scores = {}
            extended_layer = get_extend_layer(dag_circuit, front_layer, layers, node_state_dict)
            for i in range(len(swap_candidate)):
                temp_layout = layout.copy()
                # temp_layout.swap(swap_candidate[i][0], swap_candidate[i][1])temp_layout.swap(swap_candidate[i][0], swap_candidate[i][1])
                temp_layout.swap(layout.get_virtual_bits()[Qubit(qr, swap_candidate[i][0])], layout.get_virtual_bits()[Qubit(qr, swap_candidate[i][1])])
                score[i] = score_swap(qr, temp_layout, coupling_graph, front_layer, extended_layer)
            print(score)
            score_tuple = tuple(score)
            # print(type(score_tuple))
            # print(type(swap_candidate))
            for i in range(len(swap_candidate)):
                swap_scores[swap_candidate[i]] = score_tuple[i]
                # print(swap_candidate[i], swap_scores[swap_candidate[i]])
            print(swap_scores)
            min_score = min(swap_scores.values())
            # print("before swap layout:", layout)
            # print("swap_candidate_tuple ", swap_candidate)
            # print("min score:", min(swap_scores.values()))
            swap_scores_keys = list(swap_scores.keys())
            random.shuffle(swap_scores_keys)
            for item in swap_scores_keys:
                if swap_scores[item] == min_score:
                    # print(item)
                    # layout.swap(swap_scores[min(score_tuple)][0], swap_scores[min(score_tuple)][1])
                    # layout.swap(item[0], item[1])
                    layout.swap(layout.get_virtual_bits()[Qubit(qr, item[0])], layout.get_virtual_bits()[Qubit(qr, item[1])])
                    execute_schedule.append([item[0], item[1]])
                    execute_schedule.append([item[1], item[0]])
                    execute_schedule.append([item[0], item[1]])
                    print("SWAP:",[item[0], item[1]])
                    # print("swap", item[0], item[1])
                    break
            # print("after swap layout:", layout)
            # print("node state:", node_state_dict)
            # print(score)
            # print(swap_scores)
            # # print(swap_candidate)
            # print(min(score))
            # print(swap_scores[min(score)])
            # layout.swap(swap_scores[min(score)][0], swap_scores[min(score)][1])
            # break

    return layout, swap_schedule, execute_schedule


def gate_executable(qr, gate, coupling_map_list, layout):
    return [layout.get_virtual_bits()[Qubit(qr, gate[0])],
            layout.get_virtual_bits()[Qubit(qr, gate[1])]] in coupling_map_list


def get_extend_layer(dag, front_layer, layers, node_state_dict):
    extend_layer = []
    f_layer = copy.deepcopy(front_layer)
    node_state_d = copy.deepcopy(node_state_dict)
    # 可以用dag中的serial_layers()函数来重写此功能
    for i in range(layers):
        for node in f_layer:
            node_state_d[node._node_id] = 2
            # layer_identifier[node._node_id] = -1
            suc = dag.successors(node)
            for item in suc:
                if item.name == "cx":
                    pre = dag.predecessors(item)
                    for x in pre:
                        if x.name != "cx":
                            continue
                        if node_state_d[x._node_id] == 2:
                            node_state_d[item._node_id] = 3
                        else:
                            node_state_d[item._node_id] = 0
                            break
                    if node_state_d[item._node_id] == 3:
                        extend_layer.append(item)
        for node in extend_layer:
            node_state_d[node._node_id] = 2
            f_layer.append(node)
    temp = set(extend_layer)
    extend_layer = list(temp)
    return extend_layer


def score_swap(qr, layout, coupling_graph, front_layer, extend_layer):
    # , extended_layer
    sigma = 0.3
    score = 0
    score_front = 0
    score_extend = 0
    # print(front_layer)
    for item in front_layer:
        # print("count ")
        # print("qargs ", item.qargs)
        # print("gate 0 ", item.qargs[0].index)
        # print("gate 1 ", item.qargs[1].index)
        # print(layout.get_virtual_bits()[Qubit(qr, item.qargs[0].index)])
        # print(layout.get_virtual_bits()[Qubit(qr, item.qargs[1].index)])
        # print(coupling_graph.distance(layout.get_virtual_bits()[Qubit(qr, item.qargs[0].index)],
        #                          layout.get_virtual_bits()[Qubit(qr, item.qargs[1].index)]))
        score_front += coupling_graph.distance(layout.get_virtual_bits()[Qubit(qr, item.qargs[0].index)],
                                         layout.get_virtual_bits()[Qubit(qr, item.qargs[1].index)])
    score_front /= len(front_layer)
    for item in extend_layer:
        score_extend += coupling_graph.distance(layout.get_virtual_bits()[Qubit(qr, item.qargs[0].index)],
                                         layout.get_virtual_bits()[Qubit(qr, item.qargs[1].index)])
    if len(extend_layer) == 0:
        return score_front
    score_extend /= len(extend_layer)
    score = score_front*(1-sigma)+sigma*score_extend
    return score


def swap_candidate_tuple(layout, coupling_graph_list):
    swap_list = []
    for item in coupling_graph_list:
        swap_list.append((layout.get_physical_bits()[item[0]].index,
                          layout.get_physical_bits()[item[1]].index))
    return tuple(swap_list)


if __name__ == '__main__':
    # IBMQ.load_account()
    # simulated_backend = Aer.get_backend('qasm_simulator')
    # backend = IBMQ.get_provider(hub='ibm-q').get_backend('ibmq_16_melbourne')

    # distinction on these two part of getting backend.provider.coupling

    provider = IBMQ.load_account()
    backend = Aer.get_backend('qasm_simulator')
    simulated_backend = provider.get_backend('ibmq_santiago')
    # simulated_backend = provider.get_backend('ibmq_16_melbourne')
    coupling_map_list = simulated_backend.configuration().coupling_map  # Get coupling map from backend
    coupling_map = CouplingMap(coupling_map_list)

    #
    # coupling_map = [[0, 1], [0, 14], [1, 0], [1, 2], [1, 13], [2, 1], [2, 3], [2, 12], [3, 2], [3, 4], [3, 11], [4, 3],
    #                 [4, 5], [4, 10], [5, 4], [5, 6], [5, 9], [6, 5], [6, 8], [7, 8], [8, 6], [8, 7], [8, 9], [9, 5],
    #                 [9, 8], [9, 10], [10, 4], [10, 9], [10, 11], [11, 3], [11, 10], [11, 12], [12, 2], [12, 11],
    #                 [12, 13], [13, 1], [13, 12], [13, 14], [14, 0], [14, 13]]
    # coupling_map1 = CouplingMap(coupling_map)
    # print(coupling_map1.distance(12, 5))
    filename = './examples/test.qasm'
    # filename = './examples/C17_204.qasm'
    # dag = dagcircuit.DAGCircuit()



    circuit = QuantumCircuit.from_qasm_file(filename)
    circuit.measure_all()
    # new_circ_lv0 = transpile(circuit, backend=backend, optimization_level=0)
    # plot_circuit_layout(new_circ_lv0, backend).savefig('./2.png')

    qubit_num = len(circuit.qubits)
    qr = QuantumRegister(qubit_num)
    dag = converters.circuit_to_dag(circuit)
    # dag.draw()
    # plt.show()

    # for node in dag.gate_nodes():
    #     print(node._node_id)
    random_initial_layout = Layout()
    # dag.draw()
    # plt.show()
    # delete the single qubit nodes in dag circuits
    # print("nodes = ",len(dag.gate_nodes()))
    # print("nodes = ", len(dag.gate_nodes()))
    for item in dag.gate_nodes():
        # print(len(item.qargs))
        if item.name != 'cx':
            dag.remove_op_node(item)
    for item in dag.serial_layers():
        print(item)
    dag_front_layer = dag.front_layer()
    # print("nodes = ", len(dag.gate_nodes()))
    dag.draw()
    plt.show()

    # test how to get the gate from node
    # gate_list = [0, 0] * 50
    # for i in range(50):
    #     print(dag.node(i).qargs)
    #     print(dag.node(i).sort_key)
    #
    #     # print(len(dag.node(i).qargs))
    #     print(dag.node(i)._op)
    #     if dag.node(i)._op and len(dag.node(i).qargs) == 2:
    #         print(len(dag.node(i).qargs))
    #         gate_list[i] = [dag.node(i).qargs[0].index, dag.node(i).qargs[1].index]
    # print(gate_list)

    # randomized initial layout
    label = [x for x in range(qubit_num)]
    labels = random.sample(label, qubit_num)
    # print(labels)
    labelx = tuple(labels)
    # print(labelx)
    #
    # # random_initial_layout = Layout(
    # #     {(qr, 0): labelx[0], (qr, 1): labelx[1], (qr, 2): labelx[2], (qr, 3): labelx[3], (qr, 4): labelx[4],
    # #     (qr, 5): labelx[5], (qr, 6): labelx[6]})
    temp_dict = {}
    for i in range(qubit_num):
        temp_dict[qr[i]] = labelx[i]
    random_initial_layout = Layout(temp_dict)
    print("initial layout:", random_initial_layout)

    # random_initial_layout.swap(1, 2)
    # print(random_initial_layout)
    # print(random_initial_layout)
    # print(random_initial_layout.get_virtual_bits())
    # for i in random_initial_layout.get_virtual_bits():
    #     print(i)
    # print(random_initial_layout.get_virtual_bits())
    # print(random_initial_layout.get_virtual_bits().keys())

    # gate executable and get swap list
    # x = 0
    # y = 1
    # print("test ", random_initial_layout.get_virtual_bits()[Qubit(qr, x)], " ",
    #       random_initial_layout.get_virtual_bits()[Qubit(qr, y)])
    # print((random_initial_layout.get_virtual_bits()[Qubit(qr, x)],
    #       random_initial_layout.get_virtual_bits()[Qubit(qr, y)]) in coupling_map)
    # print(random_initial_layout.get_physical_bits()[0])
    # print(random_initial_layout.get_physical_bits()[1])
    # print("test ", random_initial_layout.get_physical_bits()[x].index, " ",
    #       random_initial_layout.get_physical_bits()[y].index)
    # swap_list = []
    # for item in coupling_map:
    #     swap_list.append([random_initial_layout.get_physical_bits()[item[0]].index,
    #                       random_initial_layout.get_physical_bits()[item[1]].index])
    # swap_list = [[random_initial_layout.get_physical_bits()[item[i]].index for i in range(2)]
    #                                                                   for item in coupling_map]
    # print(swap_list)
    # print(gate_executable(qr, [0, 1], coupling_map, random_initial_layout))

    # gate = [0, 1]
    # temp_gate = [0] * 2
    # for i in random_initial_layout.get_virtual_bits():
    #     print("i", i)
    #     if gate[0] == i.index:
    #         print(i.index)
    #         print(random_initial_layout.get_virtual_bits()[Qubit(qr, i.index)])
    #         temp_gate[0] = random_initial_layout.get_virtual_bits()[Qubit(qr, i.index)]
    #     if gate[1] == i.index:
    #         print(i.index)
    #         print(random_initial_layout.get_virtual_bits()[Qubit(qr, i.index)])
    #         temp_gate[1] = random_initial_layout.get_virtual_bits()[Qubit(qr, i.index)]
    # print(gate)
    # print(temp_gate)
    # print(temp_gate in coupling_map)

    '''test score calculation module '''
    # score = 0
    # print(dag_front_layer)
    # for item in dag_front_layer:
    #     # print("count ")
    #     # print("qargs ", item.qargs)
    #     # print("gate 0 ", item.qargs[0].index)
    #     # print("gate 1 ", item.qargs[1].index)
    #     # print(layout.get_virtual_bits()[Qubit(qr, item.qargs[0].index)])
    #     # print(layout.get_virtual_bits()[Qubit(qr, item.qargs[1].index)])
    #     print(coupling_map1.distance(random_initial_layout.get_virtual_bits()[Qubit(qr, item.qargs[0].index)],
    #                              random_initial_layout.get_virtual_bits()[Qubit(qr, item.qargs[1].index)]))
    #     score += coupling_map1.distance(random_initial_layout.get_virtual_bits()[Qubit(qr, item.qargs[0].index)],
    #                              random_initial_layout.get_virtual_bits()[Qubit(qr, item.qargs[1].index)])
    # print(score)
    # gate = [0, 1]
    # print(coupling_map1.distance(random_initial_layout.get_virtual_bits()[Qubit(qr, gate[0])],
    #                              random_initial_layout.get_virtual_bits()[Qubit(qr, gate[1])]))
    # print(score_swap(qr, random_initial_layout, coupling_map, dag_front_layer))

    final_layout, schedule, exe_schedule = heuristic_scheduling(qr, dag_front_layer,
                                                                coupling_map, coupling_map_list, random_initial_layout,
                                                                dag)
    print("final_layout", final_layout)
    print("execute schedule:", exe_schedule)
    # random_initial_layout, dag)
    # dag.draw()
    # plt.show()
