# https://github.com/fileaccent/graduate_paper_code/blob/682aabc740d57ec0e6bcece506423443629bcfe5/mapping/algorithm/mcpe_opt.py
from qiskit import *
from qiskit.converters import dag_to_circuit, circuit_to_dag
from collections import defaultdict
from qiskit.circuit.quantumregister import Qubit
from qiskit.dagcircuit import DAGOpNode
from copy import copy, deepcopy
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import networkx as nx
import matplotlib.pyplot as plt
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

class MCPE_opt(): # 使用CNOT门的错误率和执行时间等等
    def __init__(self, circuit, initial_mapping, backend):
        self.circuit = circuit
        self.dag = circuit_to_dag(circuit) # 量子电路的有向图
        self.initial_mapping = initial_mapping # {虚拟比特: 物理比特} 初始映射
        self.traverse_mapping = dict() # {虚拟比特: 虚拟比特} 用于添加交换门后, 后面门的应用.(因为: 一有交换门, 后面的门必须跟着换)
        self.current_mapping = None # {虚拟比特: 物理比特} 当前映射
        self.backend = backend
        self.coupling_graph = None
        self.S = None
        self.dist = None # 距离矩阵
        self.dependence_list = [] # 依赖链表
        self.g_mapping = dict() # {链表元素: 量子门}
        self._bit_indices = None # 用于交换门之间的排序
        self.bridge_num = 0
        self.swap_num = 0
    def run(self):
        self.coupling_graph = self.init_physical_graph(self.circuit, self.backend)
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
        is_not_execute_bridge = False
        bridge = {'start': 0, 'mid': 0, 'end': 0}
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
            # print('act_list_before: ', [(item.op.name, [qarg.index for qarg in item.qargs])for item in act_list])
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
                else:
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
                        control = bridge['start'].index
                        target = bridge['end'].index
                        dependence_list[control].pop(0)
                        dependence_list[target].pop(0)
                        frozen[control] = 0
                        frozen[target] = 0
                        remove_arr.append(gate)
                        # act_list.remove(gate)
                        self.bridge_num += 1
            for item in remove_arr:
                act_list.remove(item)
            candi_list = self.obtain_swap_gate(act_list)
            # print('candi_list_before: ', [[qarg.index for qarg in candi_item] for candi_item in candi_list])
            mid_candi_list = [candi_gate for candi_gate in candi_list if self.is_any_active(candi_gate, act_list)]
            if (len(mid_candi_list) > 0):
                candi_list = mid_candi_list
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
                max_MCPE_value = float('-inf')
                max_MCPE_cost = float('-inf')
                for MCPE_swap, (MCPE_value, MCPE_cost) in MCPE_costs.items():
                    if MCPE_cost > max_MCPE_cost:
                        max_swap = MCPE_swap
                        max_MCPE_value = MCPE_value
                        max_MCPE_cost = MCPE_cost
                if (max_MCPE_value <= 1): # 若交换门只对自己有益, 尝试使用桥门
                    for gate in act_list:
                        if (len(gate.qargs) == 2
                            and
                            self.S[self.current_mapping[gate.qargs[0]]][self.current_mapping[gate.qargs[1]]] == 2
                        ):
                            
                            start = gate.qargs[0]
                            end = gate.qargs[1]
                            mid = self.find_mid_qubit(start, end)
                            is_not_execute_bridge = True
                            bridge['start'] = start
                            bridge['mid'] = mid
                            bridge['end'] = end
                            break
                    if (not is_not_execute_bridge):
                        final_circuit.swap(self.traverse_mapping[max_swap[0]], self.traverse_mapping[max_swap[1]])
                        self.current_mapping = self.swap_bit_mapping(self.current_mapping, max_swap)
                        self.traverse_mapping = self.swap_bit_mapping(self.traverse_mapping, max_swap)
                        self.swap_num += 1
                else:
                    final_circuit.swap(self.traverse_mapping[max_swap[0]], self.traverse_mapping[max_swap[1]])
                    self.current_mapping = self.swap_bit_mapping(self.current_mapping, max_swap)
                    self.traverse_mapping = self.swap_bit_mapping(self.traverse_mapping, max_swap)
                    self.swap_num += 1
                # print('MCPE_cost: ', MCPE_cost)
                # print('max_swap: ', max_swap)
            if sum([len(item) for item in dependence_list]) == 0:
                break
        print('swap_num: ', self.swap_num)
        print('bridge_num: ', self.bridge_num)
        return final_circuit, self.current_mapping
    def init_physical_graph(self, circuit, backend):
        dist = self.dist = self.obtain_distance_matrix(circuit, backend)
        physical_edges = backend.configuration().coupling_map
        physical_weight_edges = []
        for item in physical_edges:
            physical_weight_edges.append((item[0], item[1], {'weight': dist[item[0]][item[1]]}))
        physical_graph = nx.Graph()
        physical_graph.add_nodes_from([i for i in range(len(backend.properties().qubits))])
        physical_graph.add_weighted_edges_from(physical_weight_edges)
        return physical_graph
    def is_executed(self, gate):
        current_mapping = self.current_mapping
        coupling_graph_edges_list = list(self.coupling_graph.edges())
        search_target = tuple([current_mapping[v] for v in gate.qargs])
        # print('coupling_graph_edges_list: ', coupling_graph_edges_list)
        # print('search_target: ', search_target)
        # print()
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
        if (gate.op.name != 'cx'): return 0, 0
        current_mapping = self.current_mapping
        diff_swap_gate_qubit = None # 查找不在CNOT_gate门中的比特
        diff_CNOT_gate_qubit = None
        diff_num = 0
        for qubit in swap_gate:
            if (not qubit in gate.qargs):
                diff_swap_gate_qubit = qubit
                diff_num += 1
        if (diff_num == 2): return 0, 0 # 说明交换门与CNOT门完全不相关
        if (diff_num == 0): return 0, 0 # 说明交换门交换控制位和目标位
        for qubit in gate.qargs:
            if (not qubit in swap_gate):
                diff_CNOT_gate_qubit = qubit
        original_qubit_0 = current_mapping[gate.qargs[0]]
        original_qubit_1 = current_mapping[gate.qargs[1]]
        original_S = self.S[original_qubit_0][original_qubit_1]
        original_dist = self.dist[original_qubit_0][original_qubit_1]
        swaped_qubit_0 = current_mapping[diff_swap_gate_qubit]
        swaped_qubit_1 = current_mapping[diff_CNOT_gate_qubit]
        swaped_S = self.S[swaped_qubit_0][swaped_qubit_1]
        swaped_dist = self.dist[swaped_qubit_0][swaped_qubit_1]
        return  original_S - swaped_S, original_dist - swaped_dist
    def is_any_active(self, swap_gate, act_list):
        for gate in act_list:
            # print('is_active: ')
            # print('swap_gate: ', swap_gate)
            # print('gate: ', gate)
            # print(self.is_active(swap_gate, gate))
            # print()
            if (self.is_active(swap_gate, gate)[0] > 0):
                return True
        return False
    def MPCE_qubit_value(self, swap_gate, qubit):
        qubit_dependence_list = self.dependence_list[qubit.index]
        MPCE_value = 0
        MPCE_cost = 0
        for dependence_item in qubit_dependence_list:
            is_active_value, is_active_cost = self.is_active(swap_gate, self.g_mapping[dependence_item])
            if (is_active_value >= 0):
                MPCE_value += is_active_value
                MPCE_cost += is_active_cost
            else:
                break
        return MPCE_value, MPCE_cost

    def MPCE_SWAP_value(self, swap_gate):
        MCPE_qubit_value_0,  MCPE_qubit_cost_0 = self.MPCE_qubit_value(swap_gate, swap_gate[0])
        MCPE_qubit_value_1,  MCPE_qubit_cost_1 = self.MPCE_qubit_value(swap_gate, swap_gate[1])
        return MCPE_qubit_value_0 + MCPE_qubit_value_1, MCPE_qubit_cost_0 + MCPE_qubit_cost_1
    def swap_bit_mapping(self, current_mapping, swap):
        tmp_mapping = deepcopy(current_mapping)
        tmp_mapping[swap[0]], tmp_mapping[swap[1]] = tmp_mapping[swap[1]], tmp_mapping[swap[0]]
        return tmp_mapping
    def find_mid_qubit(self, start, end):
        reverse_mapping = {k: v for v, k in self.current_mapping.items()}
        start = self.current_mapping[start]
        end = self.current_mapping[end]
        for i in range(len(self.dist)):
            if (self.S[start][i] == 1 and  self.S[i][end] == 1):
                return reverse_mapping[i]
        print('发生错误! 距离为2的两结点没找到中间结点!')
        return -1
    def obtain_distance_matrix (self, circuit, backend, swap_weight=0.8, error_weight=0.2, execution_time_weight=0):
        cx_message = [item.to_dict() for item in backend.properties().gates if item.gate == 'cx']
        for index, item in enumerate(cx_message):
            cx_message[index]['gate_error'] = cx_message[index]['parameters'][0]['value']
            cx_message[index]['gate_length'] = cx_message[index]['parameters'][1]['value']
            del cx_message[index]['parameters']
        # print(cx_message)
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