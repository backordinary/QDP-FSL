# https://github.com/fileaccent/graduate_paper_code/blob/682aabc740d57ec0e6bcece506423443629bcfe5/mapping/algorithm/mcpe.py
from qiskit import *
from qiskit.converters import dag_to_circuit, circuit_to_dag
from collections import defaultdict
from qiskit.circuit.quantumregister import Qubit
from qiskit.dagcircuit import DAGOpNode
from copy import copy, deepcopy
from networkx import floyd_warshall_numpy
import networkx as nx
'''
input:
    coupling_graph: 物理结构图
    initial_mapping: 初始映射
    dist: 物理结构图的距离矩阵
    dependence_list 依赖链表
ouput: final_circuit, current_mapping
定义1: active_gate: 是两比特链表前门(定义4)的门, 并且它之前的门都执行完毕
        dist: 物理比特的距离矩阵
定义2: 两比特门的最近邻距离: 两比特门所连物理比特的最近距离
定义3: 交换对两比特门的影响: 交换后最近邻距离的变化
定义4:  依赖链表: 每个比特位所使用的双比特门构成的链表
        n个比特位n个链表
        前门: 每个链表最前面的门
MCPE(SWAP, q_i): 展望窗口影响(定义3)之和
                    展望窗口: 该比特的依赖链表, 从头开始计算影响当遇到负的影响就停下, 
                            将之前的正的和无影响组成展望窗口
MCPE(SWAP) = MCPE(SWAP, q_i) + MCPE(SWAP, q_j)
'''

class MCPE():
    def __init__(self, circuit, initial_mapping, backend):
        self.circuit = circuit
        self.dag = circuit_to_dag(circuit)
        self.initial_mapping = initial_mapping # [虚拟比特: 物理比特]
        self.traverse_mapping = dict()
        self.current_mapping = None
        self.coupling_graph, self.dist = self.init_dist_coupling_graph(backend)
        self.dependence_list = []
        self.g_mapping = dict()
        self._bit_indices = None
        self.swap_num = 0
    def run(self):
        self._bit_indices = {bit: idx for idx, bit in enumerate(self.dag.qregs["q"])}
        front_list = [] # 前门不包含act_list
        act_list = [] # 可直接执行的门
        dag = self.dag
        circuit = self.circuit
        for qubit in self.dag.qubits:
            self.traverse_mapping[qubit] = qubit
        frozen = [0 for _ in range(self.coupling_graph.number_of_nodes())]
        dependence_list = [[] for _ in circuit.qubits]
        g_mapping = dict()
        for index, gate in enumerate(dag.op_nodes()):
            g_mapping[index] = gate
            for gate_qubit in gate.qargs:
                dependence_list[gate_qubit.index].append(index)
        self.g_mapping = g_mapping
        self.dependence_list = dependence_list
        final_circuit = QuantumCircuit(len(self.dag.qubits), len(self.dag.clbits))
        self.current_mapping = self.initial_mapping
        while True:
            for qubit in dag.qubits:
                if dependence_list[qubit.index]:
                    gate_key = dependence_list[qubit.index][0]
                    gate = g_mapping[gate_key]
                    is_executed = True # 表示该门的前面没有其他门
                    for qarg in gate.qargs:
                        mid_gate_key = dependence_list[qarg.index][0]
                        mid_gate = g_mapping[mid_gate_key]
                        if (mid_gate != gate):
                            is_executed = False
                    if (not is_executed and gate.op.name != 'cx'):
                        if (front_list.count(gate) == 0):
                            front_list.append(gate)
                        continue
                    if gate.op.name != 'cx':
                        for qarg in gate.qargs:
                            dependence_list[qarg.index].pop(0)
                        if (front_list.count(gate)):
                            front_list.remove(gate)
                        act_list.append(gate)
                    else:
                        if not frozen[qubit.index]:
                            if gate in front_list:
                                front_list.remove(gate)
                                act_list.append(gate)
                            else:
                                front_list.append(gate)
                            frozen[qubit.index] = 1
            remove_arr = []
            for gate in act_list:
                if gate.op.name != 'cx':
                    remove_arr.append(gate)
                    # act_list.remove(gate)
                    final_circuit.append(gate.op, [self.traverse_mapping[qarg] for qarg in gate.qargs], gate.cargs)
                    continue
                if self.is_executed(gate):
                    remove_arr.append(gate)
                    # act_list.remove(gate)
                    final_circuit.append(gate.op, [self.traverse_mapping[qarg] for qarg in gate.qargs], gate.cargs)
                    control, target = gate.qargs[0].index, gate.qargs[1].index
                    dependence_list[control].pop(0)
                    dependence_list[target].pop(0)
                    frozen[control] = 0
                    frozen[target] = 0
            for item in remove_arr:
                act_list.remove(item)
            candi_list = self.obtain_swap_gate(act_list)
            # print('candi_list_before: ', [[qarg.index for qarg in candi_item] for candi_item in candi_list])
            candi_list = [candi_gate for candi_gate in candi_list if self.is_any_active(candi_gate, act_list)]
            # print('front_list: ', [(item.op.name, [qarg.index for qarg in item.qargs])for item in front_list])
            # print('act_list: ', [(item.op.name, [qarg.index for qarg in item.qargs])for item in act_list])
            # print('candi_list: ', [[qarg.index for qarg in candi_item] for candi_item in candi_list])
            # print('final_circuit: ')
            # print(final_circuit.draw('text'))
            if len(candi_list):
                MCPE_costs = dict()
                for candi_item in candi_list:
                    MCPE_costs[candi_item] = self.MPCE_SWAP_value(candi_item)
                max_swap = None
                max_MCPE_cost = float('-inf')
                for MCPE_swap, MCPE_cost in MCPE_costs.items():
                    if MCPE_cost > max_MCPE_cost:
                        max_swap = MCPE_swap
                        max_MCPE_cost = MCPE_cost
                final_circuit.swap(self.traverse_mapping[max_swap[0]], self.traverse_mapping[max_swap[1]])
                self.current_mapping = self.swap_bit_mapping(self.current_mapping, max_swap)
                self.traverse_mapping = self.swap_bit_mapping(self.traverse_mapping, max_swap)
                self.swap_num += 1
                # print('MCPE_cost: ', MCPE_cost)
                # print('max_swap: ', max_swap)
            if sum([len(item) for item in dependence_list]) == 0:
                break
        print('swap_num: ', self.swap_num)
        return final_circuit, self.current_mapping
    def init_dist_coupling_graph(self, backend):
        coupling_graph = nx.Graph()
        edges = backend.configuration().coupling_map
        coupling_graph.add_nodes_from([i for i in range(len(backend.properties().qubits))])
        coupling_graph.add_edges_from(edges)
        distance_matrix = nx.floyd_warshall(coupling_graph)
        return coupling_graph, distance_matrix
    def is_executed(self, gate):
        current_mapping = self.current_mapping
        coupling_graph_edges_list = list(self.coupling_graph.edges())
        search_target = tuple([current_mapping[v] for v in gate.qargs])
        return coupling_graph_edges_list.count(search_target) + coupling_graph_edges_list.count(search_target[::-1])
    def obtain_swap_gate(self, front_layer):
        current_mapping = self.current_mapping
        candidate_swap_gate = set()
        traverse_current_mapping = {w: v for v, w in current_mapping.items()}
        for node in front_layer:
            if (node.op.name != 'cx'): continue
            for virtual in node.qargs:
                physical = current_mapping[virtual]
                for physical_neighbor in self.coupling_graph.neighbors(physical):
                    virtual_neighbor = traverse_current_mapping[physical_neighbor]
                    swap = sorted([virtual, virtual_neighbor], key=lambda q: self._bit_indices[q])
                    candidate_swap_gate.add(tuple(swap))
        return list(candidate_swap_gate)
    def is_active(self, swap_gate, gate): # 1 表示积极, 0 表示不相关, -1表示消极
        # 其中swap_gate 是用元组表示
        # Cgate 用DAGOpNode 表示, 有可能是单比特门
        if (gate.op.name != 'cx'): return 0
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
        original_dist = self.dist[original_qubit_0][original_qubit_1]
        swaped_qubit_0 = current_mapping[diff_swap_gate_qubit]
        swaped_qubit_1 = current_mapping[diff_CNOT_gate_qubit]
        swaped_dist = self.dist[swaped_qubit_0][swaped_qubit_1]
        return  original_dist - swaped_dist
    def is_any_active(self, swap_gate, act_list):
        for gate in act_list:
            if (self.is_active(swap_gate, gate) == 1):
                return True
        return False
    def MPCE_qubit_value(self, swap_gate, qubit):
        qubit_dependence_list = self.dependence_list[qubit.index]
        MPCE_value = 0
        for dependence_item in qubit_dependence_list:
            is_active_value = self.is_active(swap_gate, self.g_mapping[dependence_item])
            if (is_active_value >= 0):
                MPCE_value += is_active_value
            else:
                break
        return MPCE_value

    def MPCE_SWAP_value(self, swap_gate):
        return (self.MPCE_qubit_value(swap_gate, swap_gate[0])
                +
                self.MPCE_qubit_value(swap_gate, swap_gate[1])
        )
    def swap_bit_mapping(self, current_mapping, swap):
        tmp_mapping = deepcopy(current_mapping)
        tmp_mapping[swap[0]], tmp_mapping[swap[1]] = tmp_mapping[swap[1]], tmp_mapping[swap[0]]
        return tmp_mapping
