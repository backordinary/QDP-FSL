# https://github.com/fileaccent/graduate_paper_code/blob/682aabc740d57ec0e6bcece506423443629bcfe5/mapping/algorithm/sabre.py
from qiskit.converters import dag_to_circuit, circuit_to_dag
from qiskit import *
from collections import defaultdict
from copy import copy, deepcopy
from qiskit.circuit.quantumregister import Qubit
from qiskit.dagcircuit import DAGOpNode
import numpy as np
import networkx as nx
from networkx import floyd_warshall_numpy
import random
EXTENDED_SET_SIZE = 20  # Size of lookahead window. TODO: set dynamically to len(current_layout)
EXTENDED_SET_WEIGHT = 0.5  # Weight of lookahead window compared to front_layer.

DECAY_RATE = 0.001  # Decay coefficient for penalizing serial swaps.
DECAY_RESET_INTERVAL = 5  # How often to reset all decay rates to 1


class SabreSwap():
    # circuit, initial_mapping, backend
    def __init__(self, circuit, initial_mapping, backend, extended_size = 20, heuristic="decay", seed=None):
        self.circuit = circuit
        self.dag = circuit_to_dag(circuit)
        self.initial_mapping = initial_mapping
        self.traverse_mapping = dict() # 电路变换中的映射, 处理插入交换门后, 后面门的作用位置问题
        self.coupling_graph, self.distance_matrix = self.init_dist_coupling_graph(backend)
        self.applied_predecessor = defaultdict(int)
        self.heuristic = heuristic
        self._bit_indices = None
        self.seed = seed
        self.qubits_decay = None
        self.swap_num = 0
        self.extended_size = extended_size
    def run(self):
        for _, input_node in self.dag.input_map.items():
            for successor in self.obtain_successor(input_node):
                self.applied_predecessor[successor] += 1
        for qubit in self.dag.qubits:
            self.traverse_mapping[qubit] = qubit
        current_mapping = self.initial_mapping
        # 用于排列结点, 防止找到同时 {1, 3} {3, 1} 两个边
        self._bit_indices = {bit: idx for idx, bit in enumerate(self.dag.qregs["q"])}
        
        self.qubits_decay = {qubit: 1 for qubit in self.dag.qubits}

        front_layer = self.dag.front_layer()
        
        num_search_steps = 0
        final_circuit = QuantumCircuit(len(self.dag.qubits), len(self.dag.clbits))
        # 核心代码
        while front_layer:
            execute_gate_list = []
            for gate in front_layer:
                if len(gate.qargs) == 2:
                        if self.is_executed(gate, current_mapping):
                            execute_gate_list.append(gate)
                else:
                    execute_gate_list.append(gate)
            # print('front_layer: ', [(item.op.name, item.qargs) for item in front_layer])
            # print('execute_gate_list: ', [(item.op.name, item.qargs) for item in execute_gate_list])
            # print('initial_mapping: ', [(key.index, value) for key, value in self.initial_mapping.items()])
            # print('current_mapping: ', [(key.index, value) for key, value in current_mapping.items()])
            # print(self.circuit.draw('text'))
            # print(final_circuit.draw('text'))
            if execute_gate_list:
                for gate in execute_gate_list:
                    front_layer.remove(gate)
                    final_circuit.append(gate.op, [self.traverse_mapping[qarg] for qarg in gate.qargs], gate.cargs)
                    for successor in self.obtain_successor(gate):
                        self.applied_predecessor[successor] += 1
                        if self.is_resolved(successor):
                            front_layer.append(successor)
                continue
            else:
                # extended_set
                extended_set = self.obtain_extended_set(front_layer)
                swap_candidate_list = self.obtain_swap_gate(front_layer, current_mapping)
                score = dict.fromkeys(swap_candidate_list, 0)
                for swap in swap_candidate_list:
                    tmp_mapping = self.swap_bit_mapping(current_mapping, swap)
                    score[swap] = self.cost_function(tmp_mapping, front_layer, extended_set, swap)
                min_score = min(score.values())
                best_swaps = [k for k, v in score.items() if v == min_score]
                best_swaps.sort(key=lambda x: (self._bit_indices[x[0]], self._bit_indices[x[1]]))
                rng = np.random.default_rng(self.seed)
                best_swap = rng.choice(best_swaps)
                # best_swap = best_swaps[0]
                # print('candidate_list: ', [[qarg.index for qarg in item] for item in swap_candidate_list])
                # print('scores: ', score)
                # print('effect_costs: ', effect_costs)
                # print('best_gates: ', best_swaps)
                # print('best_gate: ', best_swaps)
                final_circuit.swap(*[self.traverse_mapping[item] for item in best_swap])
                self.traverse_mapping = self.swap_bit_mapping(self.traverse_mapping, best_swap)
                current_mapping = self.swap_bit_mapping(current_mapping, best_swap)
                self.swap_num += 1
            num_search_steps += 1
            if num_search_steps % DECAY_RESET_INTERVAL:
                self.reset_qubits_decay()
            else:
                self.qubits_decay[best_swap[0]] += DECAY_RATE
                self.qubits_decay[best_swap[1]] += DECAY_RATE
        print('swap_num: ', self.swap_num)
        return final_circuit, self.initial_mapping

    def is_executed(self, gate, current_mapping):
        coupling_graph_edges_list = list(self.coupling_graph.edges())
        search_target = tuple([current_mapping[v] for v in gate.qargs])
        return coupling_graph_edges_list.count(search_target) +  + coupling_graph_edges_list.count(search_target[::-1])
    def init_dist_coupling_graph(self, backend):
        coupling_graph = nx.Graph()
        coupling_graph.add_nodes_from([i for i in range(len(backend.properties().qubits))])
        edges = backend.configuration().coupling_map
        coupling_graph.add_edges_from(edges)
        distance_matrix = floyd_warshall_numpy(coupling_graph)
        # print(distance_matrix)
        return coupling_graph, distance_matrix
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
                        if len(successor.qargs) == 2:
                            extended_set.append(successor)
                if len(extended_set) >= self.extended_size:
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
    def cost_function(self, tmp_mapping, front_layer, extended_set, swap_qubits):
        first_cost = 0
        for node in front_layer:
            first_cost += self.distance_matrix[tmp_mapping[node.qargs[0]]][tmp_mapping[node.qargs[1]]]
        
        if self.heuristic == 'basic':
            return first_cost
        
        first_cost /= len(front_layer)
        second_cost = 0
        if extended_set:
            for node in extended_set:
                second_cost += self.distance_matrix[tmp_mapping[node.qargs[0]]][tmp_mapping[node.qargs[1]]]
            second_cost /= len(extended_set)
        total_cost = first_cost + EXTENDED_SET_WEIGHT * second_cost
        if self.heuristic == 'lookahead':
            return total_cost
        if self.heuristic == 'decay':
            return (
                max(self.qubits_decay[swap_qubits[0]], self.qubits_decay[swap_qubits[1]])
                * total_cost
            )
        return first_cost
    def swap_bit_mapping(self, current_mapping, swap):
        tmp_mapping = deepcopy(current_mapping)
        tmp_mapping[swap[0]], tmp_mapping[swap[1]] = tmp_mapping[swap[1]], tmp_mapping[swap[0]]
        return tmp_mapping
    def reset_qubits_decay(self):
        self.qubits_decay = {qubit: 1 for qubit in self.dag.qubits}

class Sabre():
    def __init__(self, circuit, backend, extended_size=20, heuristic="decay", seed=None, max_iteration=1):
        self.dag = circuit_to_dag(circuit)
        self.backend = backend
        self.applied_predecessor = defaultdict(int)
        self.heuristic = heuristic
        self.max_iteration  = max_iteration
        self.extended_size = extended_size
    def run(self):
        circ = dag_to_circuit(self.dag)
        rev_circ = circ.reverse_ops()
        initial_mapping = self.random_initial_mapping(circ)
        sabre_proc = None
        new_circ = None
        all_mapping = set()
        all_mapping.add(''.join(['{}{}'.format(key.index, value) for key, value in initial_mapping.items()]))
        for i in range(self.max_iteration):
            sabre_proc = SabreSwap(circ, initial_mapping, self.backend, self.extended_size)
            new_circ, final_mapping = sabre_proc.run()
            initial_mapping = final_mapping
            circ, rev_circ = rev_circ, circ
            all_mapping.add(''.join(['{}{}'.format(key.index, value) for key, value in initial_mapping.items()]))
        
        all_mapping = list(all_mapping)
        all_mapping.sort()
        # '_'.join(all_mapping)
        return new_circ, final_mapping
    def random_initial_mapping(self, circuit):
        # test_mapping =[
        # (0, 0), (1, 3), (2, 6), (3, 11), (4, 8), 
        # (5, 10), (6, 5), (7, 4), (8, 15), (9, 12), 
        # (10, 14), (11, 7), (12, 1), (13, 9), (14, 13),
        # (15, 2)
        # ]
        # return {circuit.qubits[item[0]]: item[1] for item in test_mapping}
        initial_mapping = dict()
        circuit_qubits = deepcopy(circuit.qubits)
        random.shuffle(circuit_qubits)
        for index, qubit in enumerate(circuit_qubits):
            initial_mapping[qubit] = index
        return initial_mapping