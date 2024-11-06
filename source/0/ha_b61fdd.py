# https://github.com/fileaccent/graduate_paper_code/blob/682aabc740d57ec0e6bcece506423443629bcfe5/mapping/algorithm/ha.py
from qiskit.converters import dag_to_circuit, circuit_to_dag
from qiskit import *
from qiskit.test.mock import FakeMontreal
import networkx as nx
from collections import defaultdict
from copy import copy, deepcopy
from qiskit.circuit.quantumregister import Qubit
from qiskit.dagcircuit import DAGOpNode
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
from sympy import false, true
import pandas as pd
import matplotlib.pyplot as plt
'''
    S矩阵: 普通的距离矩阵
    E矩阵: 误差矩阵
    T矩阵: 执行时间矩阵
'''

EXTENDED_SET_SIZE = 20  # Size of lookahead window. TODO: set dynamically to len(current_layout)
EXTENDED_SET_WEIGHT = 0.5  # Weight of lookahead window compared to front_layer.

DECAY_RATE = 0.001  # Decay coefficient for penalizing serial swaps.
DECAY_RESET_INTERVAL = 5  # How often to reset all decay rates to 1

class HA_paper ():
    def __init__ (self, circuit, initial_mapping, backend, heuristic="decay", seed=None):
        self.circuit = circuit
        self.dag = circuit_to_dag(circuit)
        self.initial_mapping = initial_mapping
        self.traverse_mapping = dict()
        self.backend = backend
        self.distance_matrix = []
        self.S = []
        self.coupling_graph = None
        self.applied_predecessor = defaultdict(int)
        self.heuristic = heuristic
        self._bit_indices = None
        self.seed = seed
        self.qubits_decay = None
        self.swap_num = 0
        self.bridge_num = 0
    def run(self):
        # nx.draw(self.coupling_graph, with_labels=True)
        # plt.show()
        self.distance_matrix = self.obtain_distance_matrix(self.backend, 0.8, 0.2, 0)
        for _, input_node in self.dag.input_map.items():
            for successor in self.obtain_successor(input_node):
                self.applied_predecessor[successor] += 1
        for qubit in self.dag.qubits:
            self.traverse_mapping[qubit] = qubit
        self.current_mapping = self.initial_mapping
        # 用于排列结点, 防止找到同时 {1, 3} {3, 1} 两个边
        self._bit_indices = {bit: idx for idx, bit in enumerate(self.dag.qregs["q"])}
        
        self.qubits_decay = {qubit: 1 for qubit in self.dag.qubits}

        front_layer = self.dag.front_layer()
        final_circuit = QuantumCircuit(len(self.dag.qubits), len(self.dag.clbits))
        is_not_execute_bridge = False
        bridge = {'start': 0, 'mid': 0, 'end': 0}
        # 核心代码
        while front_layer:
            execute_gate_list = []
            for gate in front_layer:
                if (gate.op.name == 'cx'):
                    if self.is_executed(gate, self.current_mapping):
                        execute_gate_list.append(gate)
                    elif (is_not_execute_bridge and bridge['start'] == gate.qargs[0] and bridge['end'] == gate.qargs[1]):
                        execute_gate_list.append(gate)
                else:
                    execute_gate_list.append(gate)
            print('front_layer: ', [(item.op.name, [qarg.index for qarg in item.qargs]) for item in front_layer])
            print('execute_gate_list: ', execute_gate_list)
            print(final_circuit.draw('text'))
            if execute_gate_list:
                for gate in execute_gate_list:
                    front_layer.remove(gate)
                    if is_not_execute_bridge and bridge['start'] == gate.qargs[0] and bridge['end'] == gate.qargs[1]:
                        start = self.traverse_mapping[bridge['start']]
                        mid = self.traverse_mapping[bridge['mid']]
                        end = self.traverse_mapping[bridge['end']]
                        final_circuit.barrier()
                        final_circuit.cx(mid, end)
                        final_circuit.cx(start, mid)
                        final_circuit.cx(mid, end)
                        final_circuit.cx(start, mid)
                        final_circuit.barrier()
                        is_not_execute_bridge = False
                    else:
                        final_circuit.append(gate.op, [self.traverse_mapping[qarg] for qarg in gate.qargs], gate.cargs)
                        # getattr(final_circuit, gate.op.name)(*[self.traverse_mapping[qarg] for qarg in gate.qargs])
                    for successor in self.obtain_successor(gate):
                        self.applied_predecessor[successor] += 1
                        if self.is_resolved(successor):
                            front_layer.append(successor)
            else:
                # extended_set
                extended_set = self.obtain_extended_set(front_layer)
                swap_candidate_list = self.obtain_swap_gate(front_layer, self.current_mapping)
                score = dict.fromkeys(swap_candidate_list, 0)
                effect_costs = dict.fromkeys(swap_candidate_list, 0)
                for swap in swap_candidate_list:
                    tmp_mapping = self.swap_bit_mapping(self.current_mapping, swap)
                    score[swap], effect_costs[swap] = self.cost_function(tmp_mapping, front_layer, extended_set, swap)
                min_score = min(score.values())
                best_gates = [k for k, v in score.items() if v == min_score]
                best_gates.sort(key=lambda x: (self._bit_indices[x[0]], self._bit_indices[x[1]]))
                # rng = np.random.default_rng(self.seed)
                # best_gate = rng.choice(best_gates)
                best_gate = best_gates[0]
                print('score: ', score)
                print('best_gates: ', best_gates)
                if (effect_costs[best_gate] < 0 and self.S[self.current_mapping[best_gate[0]]][self.current_mapping[best_gate[1]]] == 2): # 添加桥门
                    start = best_gate[0]
                    end = best_gate[1]
                    mid = self.find_mid_qubit(self.current_mapping[best_gate[0]], self.current_mapping[best_gate[1]])
                    bridge['start'] = start
                    bridge['mid'] = mid
                    bridge['end'] = end
                    is_not_execute_bridge = True
                    self.bridge_num += 1
                else:
                    final_circuit.swap(*[self.traverse_mapping[item] for item in best_gate])
                    self.traverse_mapping = self.swap_bit_mapping(self.traverse_mapping, best_gate)
                    self.current_mapping = self.swap_bit_mapping(self.current_mapping, best_gate)
                    self.swap_num += 1
        print('bridge_num: ', self.bridge_num)
        print('swap_num: ', self.swap_num)
        return final_circuit, self.current_mapping

    def is_executed(self, gate, current_mapping):
        coupling_graph_edges_list = list(self.coupling_graph.edges())
        search_target = tuple([current_mapping[v] for v in gate.qargs])
        return coupling_graph_edges_list.count(search_target) +  + coupling_graph_edges_list.count(search_target[::-1])
    def obtain_successor(self, gate):
        for _, successor, edge_data in self.dag.edges(gate):
            if not isinstance(successor, DAGOpNode):
                continue
            if isinstance(edge_data, Qubit):
                yield successor
    def obtain_extended_set(self, front_layer):
        extended_set = []
        incremented = []
        tmp_front_layer = front_layer
        done = False
        while tmp_front_layer and not done:
            new_tmp_front_layer = []
            for node in tmp_front_layer:
                for successor in self.obtain_successor(node):
                    incremented.append(successor)
                    self.applied_predecessor[successor] += 1
                    if self.is_resolved(successor):
                        new_tmp_front_layer.append(successor)
                        if len(successor.qargs) == 2 and successor.op.name == 'cx':
                            extended_set.append(successor)
                if len(extended_set) >= EXTENDED_SET_SIZE:
                    done = True
                    break
            tmp_front_layer = new_tmp_front_layer
        for node in incremented:
            self.applied_predecessor[node] -= 1
        return extended_set
    def is_resolved(self, successor):
        return self.applied_predecessor[successor] == len(successor.qargs)
    def obtain_swap_gate(self, front_layer, current_mapping):
        candidate_swap_gate = set()
        traverse_current_mapping = {w: v for v, w in current_mapping.items()}
        for node in front_layer:
            for virtual in node.qargs:
                physical = current_mapping[virtual]
                for physical_neighbor in self.coupling_graph.neighbors(physical):
                    virtual_neighbor = traverse_current_mapping[physical_neighbor]
                    swap = sorted([virtual, virtual_neighbor], key=lambda q: self._bit_indices[q])
                    candidate_swap_gate.add(tuple(swap))
        return candidate_swap_gate
    def cost_function(self, tmp_mapping, front_layer, extended_set, gate):

        first_cost = 0
        effect_costs = 0
        for node in front_layer:
            if (node.op.name != 'cx'): continue
            first_cost += self.distance_matrix[tmp_mapping[node.qargs[0]]][tmp_mapping[node.qargs[1]]]
            if(len(gate) == 3 and tmp_mapping[gate[0]] == tmp_mapping[node.qargs[0]] and tmp_mapping[gate[2]] == tmp_mapping[node.qargs[1]]):
                first_cost -= 1
        if self.heuristic == 'basic':
            return first_cost, effect_costs
        
        first_cost /= len(front_layer)
        second_cost = 0
        if extended_set:
            for node in extended_set:
                origin_distance = self.distance_matrix[self.current_mapping[node.qargs[0]]][self.current_mapping[node.qargs[1]]]
                swaped_distance = self.distance_matrix[tmp_mapping[node.qargs[0]]][tmp_mapping[node.qargs[1]]]
                second_cost += swaped_distance
                effect_costs += origin_distance - swaped_distance
            second_cost /= len(extended_set)
        total_cost = first_cost + EXTENDED_SET_WEIGHT * second_cost
        if self.heuristic == 'lookahead':
            return total_cost, effect_costs
        if self.heuristic == 'decay':
            return (
                max(self.qubits_decay[gate[0]], self.qubits_decay[gate[1]])
                * total_cost
            ), effect_costs
        return first_cost, effect_costs
    def swap_bit_mapping(self, current_mapping, swap):
        tmp_mapping = deepcopy(current_mapping)
        tmp_mapping[swap[0]], tmp_mapping[swap[1]] = tmp_mapping[swap[1]], tmp_mapping[swap[0]]
        return tmp_mapping
    def reset_qubits_decay(self):
        self.qubits_decay = {qubit: 1 for qubit in self.dag.qubits}
    def obtain_distance_matrix (self, backend, swap_weight, error_weight, execution_time_weight):
        cx_message = [item.to_dict() for item in backend.properties().gates if item.gate == 'cx']
        for index, item in enumerate(cx_message):
            cx_message[index]['gate_error'] = cx_message[index]['parameters'][0]['value']
            cx_message[index]['gate_length'] = cx_message[index]['parameters'][1]['value']
            del cx_message[index]['parameters']
        qubit_num = len(backend.properties().qubits)
        max_limit = qubit_num ** 3 * 500
        S = [[max_limit for _ in range(qubit_num)] for _ in range(qubit_num)]
        E = [[max_limit for _ in range(qubit_num)] for _ in range(qubit_num)]
        T = [[max_limit for _ in range(qubit_num)] for _ in range(qubit_num)]
        for i in range(len(S)):
            S[i][i] = E[i][i] = T[i][i] = 0
        coupling_graph = nx.Graph()
        coupling_graph.add_nodes_from([i for i in range(len(backend.properties().qubits))])
        for item in cx_message:
            coupling_graph.add_edge(
                item['qubits'][0], 
                item['qubits'][1], 
                gate_error=item['gate_error'], 
                gate_length=item['gate_length'],
                weight = 1
            )
            i = item['qubits'][0]
            j = item['qubits'][1]
            S[i][j] = 1
            E[i][j] = item['gate_error']
            T[i][j] = item['gate_length']
        for i in range(len(E)):
            for j in range(len(E[i])):
                if (E[i][j] < max_limit):
                    E[i][j] = 1 - E[i][j] * E[j][i] * max(E[i][j], E[j][i])
                if (T[i][j] < max_limit):
                    T[i][j] = T[i][j] + T[j][i] + min(T[i][j], T[j][i])
        S = self.floyd(S)
        self.S = deepcopy(S)
        E = self.floyd(E)
        T = self.floyd(T)
        S = self.minMaxScaler(S, max_limit)
        E = self.minMaxScaler(E, max_limit)
        T = self.minMaxScaler(T, max_limit)
        for i in range(len(S)):
            for j in range(len(S)):
                S[i][j] *= swap_weight
                E[i][j] *= error_weight
                T[i][j] *= execution_time_weight
        self.coupling_graph = coupling_graph
        return S + E + T
    def floyd(self, M):
        for i in range(len(M)):
            for j in range(len(M)):
                for k in range(len(M)):
                    if (M[i][j] > M[i][k] + M[k][j]):
                        M[i][j] = M[i][k] + M[k][j]
        return M
    def minMaxScaler(self, M, MAX_LIMIT):
        min_value = float('inf')
        max_value = float('-inf')
        for i in range(len(M)):
            for j in range(len(M)):
                if (M[i][j] <= MAX_LIMIT):
                    if (M[i][j] > max_value):
                        max_value = M[i][j]
                    if (M[i][j] < min_value):
                        min_value = M[i][j]
        for i in range(len(M)):
            for j in range(len(M)):
                if (M[i][j] <= MAX_LIMIT):
                    M[i][j] = (M[i][j] - min_value) / (max_value - min_value)
        return M
    def find_mid_qubit(self, start, end):
        for i in range(len(self.S)):
            if (self.S[start][i] == 1 and  self.S[i][end] == 1):
                return i
        print('发生错误! 距离为2的两结点没找到中间结点!')
        return -1

class HA ():
    def __init__ (self, circuit, initial_mapping, backend, heuristic="decay", seed=None):
        self.circuit = circuit
        self.dag = circuit_to_dag(circuit)
        self.initial_mapping = initial_mapping
        self.traverse_mapping = dict()
        self.backend = backend
        self.distance_matrix = []
        self.S = []
        self.coupling_graph = None
        self.applied_predecessor = defaultdict(int)
        self.heuristic = heuristic
        self._bit_indices = None
        self.seed = seed
        self.qubits_decay = None
        self.swap_num = 0
        self.bridge_num = 0
    def run(self):
        self.distance_matrix = self.obtain_distance_matrix(self.backend, 0.8, 0.2, 0)
        # nx.draw(self.coupling_graph, with_labels=True)
        # plt.show()
        for _, input_node in self.dag.input_map.items():
            for successor in self.obtain_successor(input_node):
                self.applied_predecessor[successor] += 1
        for qubit in self.dag.qubits:
            self.traverse_mapping[qubit] = qubit
        self.current_mapping = self.initial_mapping
        # 用于排列结点, 防止找到同时 {1, 3} {3, 1} 两个边
        self._bit_indices = {bit: idx for idx, bit in enumerate(self.dag.qregs["q"])}
        
        self.qubits_decay = {qubit: 1 for qubit in self.dag.qubits}
        
        front_layer = self.dag.front_layer()
        
        num_search_steps = 0

        final_circuit = QuantumCircuit(len(self.dag.qubits), len(self.dag.clbits))
        # 核心代码
        before_best_gate = None
        is_not_execute_bridge = False
        bridge = {'start': 0, 'mid': 0, 'end': 0}
        while front_layer:
            execute_gate_list = []
            for gate in front_layer:
                if (gate.op.name == 'cx'):
                    if self.is_executed(gate, self.current_mapping):
                        execute_gate_list.append(gate)
                    elif (is_not_execute_bridge and bridge['start'] == gate.qargs[0] and bridge['end'] == gate.qargs[1]):
                        execute_gate_list.append(gate)
                else:
                    execute_gate_list.append(gate)
            # print('front_layer: ', [(item.op.name, item.qargs) for item in front_layer])
            # print('execute_gate_list: ', [(item.op.name, item.qargs) for item in execute_gate_list])
            # print('initial_mapping: ', [(key.index, value) for key, value in self.initial_mapping.items()])
            # print('current_mapping: ', [(key.index, value) for key, value in self.current_mapping.items()])
            # print(self.circuit.draw('text'))
            # print(final_circuit.draw('text'))
            if execute_gate_list:
                for gate in execute_gate_list:
                    front_layer.remove(gate)
                    if is_not_execute_bridge and bridge['start'] == gate.qargs[0] and bridge['end'] == gate.qargs[1]:
                        start = self.traverse_mapping[bridge['start']]
                        mid = self.traverse_mapping[bridge['mid']]
                        end = self.traverse_mapping[bridge['end']]
                        final_circuit.barrier()
                        final_circuit.cx(mid, end)
                        final_circuit.cx(start, mid)
                        final_circuit.cx(mid, end)
                        final_circuit.cx(start, mid)
                        final_circuit.barrier()
                        is_not_execute_bridge = False
                    else:
                        final_circuit.append(gate.op, [self.traverse_mapping[qarg] for qarg in gate.qargs], gate.cargs)
                        # getattr(final_circuit, gate.op.name)(*[self.traverse_mapping[qarg] for qarg in gate.qargs])
                    for successor in self.obtain_successor(gate):
                        self.applied_predecessor[successor] += 1
                        if self.is_resolved(successor):
                            front_layer.append(successor)
            else:
                # extended_set
                extended_set = self.obtain_extended_set(front_layer)
                swap_candidate_list = self.obtain_swap_gate(front_layer, self.current_mapping)
                # 过滤候选门至少剩一个门
                # mid_swap_candidate_list = [item for item in swap_candidate_list if self.is_any_active_swap(front_layer, item)]
                # swap_candidate_list = mid_swap_candidate_list
                bridge_candidate_list = self.obtain_bridge_gate(front_layer, self.current_mapping)
                # 过滤候选门至少剩一个门
                # mid_bridge_candidate_list = [item for item in bridge_candidate_list if self.is_any_active_bridge(front_layer, item)]
                # if (len(mid_bridge_candidate_list)): bridge_candidate_list = mid_bridge_candidate_list
                candidate_list = swap_candidate_list + bridge_candidate_list
                scores = dict.fromkeys(candidate_list, 0)
                effect_costs = dict.fromkeys(candidate_list, 0)
                for gate in candidate_list:
                    if len(list(gate)) == 2: # 交换门
                        tmp_mapping = self.swap_bit_mapping(self.current_mapping, gate)
                        scores[gate], effect_costs[gate] = self.cost_function(tmp_mapping, front_layer, extended_set, gate)
                    elif len(list(gate)) == 3:
                        scores[gate], effect_costs[gate] = self.cost_function(self.current_mapping, front_layer, extended_set, gate)
                    else:
                        print('门的数量出错!为', len(list(gate)))
                min_effect_costs = min(effect_costs.values())
                best_gates = [k for k, v in effect_costs.items() if v == min_effect_costs]
                # min_costs = min([scores[item] for item in best_gates])
                # best_gates = [item for item in best_gates if scores[item] == min_costs]
                def compare(x):
                    if (len(list(x))) == 2:
                        return (self._bit_indices[x[0]], self._bit_indices[x[1]])
                    elif (len(list(x))) == 3:
                        return (self._bit_indices[x[0]], self._bit_indices[x[1]], self._bit_indices[x[2]])
                best_gates.sort(key=compare)
                rng = np.random.default_rng(self.seed)
                best_gate = rng.choice(best_gates)
                # print('candidate_list: ', [[qarg.index for qarg in item] for item in candidate_list])
                # print('scores: ', scores)
                # print('effect_costs: ', effect_costs)
                # print('best_gates: ', best_gates)
                # print('best_gate: ', best_gate)
                best_gate = best_gates[0]
                if (len(list(best_gate)) == 3): # 添加桥门
                    bridge['start'] = best_gate[0]
                    bridge['mid'] = best_gate[1]
                    bridge['end'] = best_gate[2]
                    is_not_execute_bridge = True
                    self.bridge_num += 1
                elif (len(list(best_gate)) == 2):
                    final_circuit.swap(*[self.traverse_mapping[item] for item in best_gate])
                    self.traverse_mapping = self.swap_bit_mapping(self.traverse_mapping, best_gate)
                    self.current_mapping = self.swap_bit_mapping(self.current_mapping, best_gate)
                    self.swap_num += 1
                else:
                    print('门的数量发生错误!为: ', len(list(best_gate)))
                num_search_steps += 1
                if num_search_steps % DECAY_RESET_INTERVAL:
                    self.reset_qubits_decay()
                else:
                    self.qubits_decay[best_gate[0]] += DECAY_RATE
                    self.qubits_decay[best_gate[1]] += DECAY_RATE
        print('bridge_num: ', self.bridge_num)
        print('swap_num: ', self.swap_num)
        return final_circuit, self.current_mapping

    def is_executed(self, gate, current_mapping):
        coupling_graph_edges_list = list(self.coupling_graph.edges())
        search_target = tuple([current_mapping[v] for v in gate.qargs])
        return coupling_graph_edges_list.count(search_target) +  + coupling_graph_edges_list.count(search_target[::-1])
    def obtain_successor(self, gate):
        for _, successor, edge_data in self.dag.edges(gate):
            if not isinstance(successor, DAGOpNode):
                continue
            if isinstance(edge_data, Qubit):
                yield successor
    def obtain_extended_set(self, front_layer):
        extended_set = []
        incremented = []
        tmp_front_layer = front_layer
        done = False
        while tmp_front_layer and not done:
            new_tmp_front_layer = []
            for node in tmp_front_layer:
                for successor in self.obtain_successor(node):
                    incremented.append(successor)
                    self.applied_predecessor[successor] += 1
                    if self.is_resolved(successor):
                        new_tmp_front_layer.append(successor)
                        if len(successor.qargs) == 2 and successor.op.name == 'cx':
                            extended_set.append(successor)
                if len(extended_set) >= EXTENDED_SET_SIZE:
                    done = True
                    break
            tmp_front_layer = new_tmp_front_layer
        for node in incremented:
            self.applied_predecessor[node] -= 1
        return extended_set
    def is_resolved(self, successor):
        return self.applied_predecessor[successor] == len(successor.qargs)
    def obtain_swap_gate(self, front_layer, current_mapping):
        candidate_swap_gate = set()
        traverse_current_mapping = {w: v for v, w in current_mapping.items()}
        for node in front_layer:
            for virtual in node.qargs:
                physical = current_mapping[virtual]
                for physical_neighbor in self.coupling_graph.neighbors(physical):
                    # if not physical_neighbor in traverse_current_mapping.keys(): continue
                    virtual_neighbor = traverse_current_mapping[physical_neighbor]
                    swap = sorted([virtual, virtual_neighbor], key=lambda q: self._bit_indices[q])
                    candidate_swap_gate.add(tuple(swap))
        return list(candidate_swap_gate)
    def cost_function(self, tmp_mapping, front_layer, extended_set, gate):

        first_cost = 0
        first_S = 0
        for node in front_layer:
            if (node.op.name != 'cx'): continue
            first_cost += self.distance_matrix[tmp_mapping[node.qargs[0]]][tmp_mapping[node.qargs[1]]]
            first_S += self.S[tmp_mapping[node.qargs[0]]][tmp_mapping[node.qargs[1]]]
            if(len(gate) == 3 and tmp_mapping[gate[0]] == tmp_mapping[node.qargs[0]] and tmp_mapping[gate[2]] == tmp_mapping[node.qargs[1]]):
                first_cost -= 1
        first_cost /= len(front_layer)
        first_S /= len(front_layer)
        if self.heuristic == 'basic':
            return first_cost, first_S
        second_cost = 0
        second_S = 0
        if extended_set:
            for node in extended_set:
                origin_distance = self.distance_matrix[self.current_mapping[node.qargs[0]]][self.current_mapping[node.qargs[1]]]
                swaped_distance = self.distance_matrix[tmp_mapping[node.qargs[0]]][tmp_mapping[node.qargs[1]]]
                second_cost += swaped_distance
                second_S += self.S[tmp_mapping[node.qargs[0]]][tmp_mapping[node.qargs[1]]]
            second_cost /= len(extended_set)
            second_S /= len(extended_set)
        total_cost = first_cost + EXTENDED_SET_WEIGHT * second_cost
        total_S = first_S + EXTENDED_SET_WEIGHT * second_S
        if self.heuristic == 'lookahead':
            return total_cost, total_S
        if self.heuristic == 'decay':
            return (
                max(self.qubits_decay[gate[0]], self.qubits_decay[gate[1]])
                * total_cost
            ),  (
                max(self.qubits_decay[gate[0]], self.qubits_decay[gate[1]])
                * total_S
            )
        return first_cost, first_S
    def swap_bit_mapping(self, current_mapping, swap):
        tmp_mapping = deepcopy(current_mapping)
        tmp_mapping[swap[0]], tmp_mapping[swap[1]] = tmp_mapping[swap[1]], tmp_mapping[swap[0]]
        return tmp_mapping
    def reset_qubits_decay(self):
        self.qubits_decay = {qubit: 1 for qubit in self.dag.qubits}
    def obtain_distance_matrix (self, backend, swap_weight, error_weight, execution_time_weight):
        cx_message = [item.to_dict() for item in backend.properties().gates if item.gate == 'cx']
        for index, item in enumerate(cx_message):
            cx_message[index]['gate_error'] = cx_message[index]['parameters'][0]['value']
            cx_message[index]['gate_length'] = cx_message[index]['parameters'][1]['value']
            del cx_message[index]['parameters']
        qubit_num = len(backend.properties().qubits)
        S = [[0 for _ in range(qubit_num)] for _ in range(qubit_num)]
        E = [[0 for _ in range(qubit_num)] for _ in range(qubit_num)]
        T = [[0 for _ in range(qubit_num)] for _ in range(qubit_num)]
        for i in range(len(S)):
            S[i][i] = E[i][i] = T[i][i] = 0
        coupling_graph = nx.Graph()
        coupling_graph.add_nodes_from([i for i in range(len(backend.properties().qubits))])
        for item in cx_message:
            coupling_graph.add_edge(
                item['qubits'][0], 
                item['qubits'][1], 
                gate_error=item['gate_error'], 
                gate_length=item['gate_length'],
                weight = 1
            )
            i = item['qubits'][0]
            j = item['qubits'][1]
            S[i][j] = 1
            E[i][j] = item['gate_error']
            T[i][j] = item['gate_length']
        # print('coupling_graph: ', coupling_graph.adj)
        for i in range(len(E)):
            for j in range(len(E[i])):
                    E[i][j] = 1 - E[i][j] * E[j][i] * max(E[i][j], E[j][i])
                    T[i][j] = T[i][j] + T[j][i] + min(T[i][j], T[j][i])
        for (i, j) in coupling_graph.edges():
            coupling_graph[i][j]['weight'] = 1
            coupling_graph[i][j]['gate_error'] = E[i][j]
            coupling_graph[i][j]['gate_length'] = T[i][j]
        S = nx.floyd_warshall_numpy(coupling_graph, weight='weight')
        self.S = deepcopy(S)
        E = nx.floyd_warshall_numpy(coupling_graph, weight='gate_error')
        T = nx.floyd_warshall_numpy(coupling_graph, weight='gate_length')
        S = MinMaxScaler().fit_transform(S)
        E = MinMaxScaler().fit_transform(E)
        T = MinMaxScaler().fit_transform(T)
        for i in range(len(S)):
            for j in range(len(S)):
                S[i][j] *= swap_weight
                E[i][j] *= error_weight
                T[i][j] *= execution_time_weight
        self.coupling_graph = coupling_graph
        return S + E + T
    def find_mid_qubit(self, start, end):
        for i in range(len(self.S)):
            if (self.S[start][i] == 1 and  self.S[i][end] == 1):
                return i
        print('发生错误! 距离为2的两结点没找到中间结点!')
        return -1
    def obtain_bridge_gate(self, front_layer, current_mapping):
        candidate_bridge_gate = set()
        traverse_current_mapping = {w: v for v, w in current_mapping.items()}
        for node in front_layer:
            for start_virtual in node.qargs:
                start_physical = current_mapping[start_virtual]
                for mid_physical in self.coupling_graph.neighbors(start_physical):
                    # if not mid_physical in traverse_current_mapping.keys(): continue
                    mid_virtual = traverse_current_mapping[mid_physical]
                    for end_physical in self.coupling_graph.neighbors(mid_physical):
                        # if not end_physical in traverse_current_mapping.keys(): continue
                        end_virtual = traverse_current_mapping[end_physical]
                        bridge = [start_virtual, mid_virtual, end_virtual]
                        if start_virtual != end_virtual and self.is_any_active_bridge(front_layer, bridge):
                            candidate_bridge_gate.add(tuple(bridge))
        return list(candidate_bridge_gate)
    def is_any_active_bridge(self, front_layer, bridge):
        for gate in front_layer:
            if (len(gate.qargs) == 2 and bridge[0] == gate.qargs[0] and bridge[2] == gate.qargs[1]):
                return true
        return false
    def is_any_active_swap(self, front_layer, swap):
        for gate in front_layer:
            if (len(gate.qargs) == 2 and self.is_active(swap, gate)):
                return true
        return false
    def is_active(self, swap_gate, gate): # 1 表示积极, 0 表示不相关, -1表示消极
        # 其中swap_gate 是用元组表示
        # Cgate 用DAGOpNode 表示, 有可能是单比特门
        if (len(gate.qargs) == 1): return 0
        current_mapping = self.current_mapping
        diff_swap_gate_qubit = None # 查找不在CNOT_gate门中的比特
        diff_CNOT_gate_qubit = None
        diff_num = 0
        for qubit in swap_gate:
            if (not qubit in gate.qargs):
                diff_swap_gate_qubit = qubit
                diff_num += 1
        if (diff_num == 2): return 0 # 说明交换门与CNOT门完全不相关
        if (diff_num == 0): return 0 # 说明交换门交换控制位和目标位
        for qubit in gate.qargs:
            if (not qubit in swap_gate):
                diff_CNOT_gate_qubit = qubit
        original_qubit_0 = current_mapping[gate.qargs[0]]
        original_qubit_1 = current_mapping[gate.qargs[1]]
        original_dist = self.distance_matrix[original_qubit_0][original_qubit_1]
        swaped_dist = self.distance_matrix[diff_swap_gate_qubit.index][diff_CNOT_gate_qubit.index]
        return  original_dist - swaped_dist