# https://github.com/hflash/profiling/blob/0498dff8c1901591d4428c3149eb3aadf80ac483/circuittransform/inputs/inputgenerator.py
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 11:53:26 2019

@author: Xiangzhen Zhou
"""
import copy
import networkx as nx
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from circuittransform.inputs.operationU import OperationCNOT
import circuittransform as ct
import os
#from qiskit.extensions import standard
from qiskit.converters import circuit_to_dag
from circuittransform.inputs.operationU import OperationCNOT, OperationU3, OperationSingle
#from circuittransform import OperationCNOT

def CreateCNOTRandomly(q_reg, num_CNOT, cir = None):
    '''
    generate CNOT operation randomly
    input:
        q_reg: quantum register
        cir: if have, input generated operations to this quantum circuit
    return:
        list of all operatuions
    '''
    
    q = q_reg
    num_qubits = len(q)
    # store all CNOT operation instances
    total_CNOT = []
    # seed for generating random input qubits for each CNOT operations
    seed = np.array(range(num_qubits))
    CNOT_input = np.zeros((num_CNOT, 2))
    for i in range(num_CNOT):
        np.random.shuffle(seed)
        CNOT_input[i, 0:2] = seed[0:2]
    '''generate OperationCNOT instances assuming no swap of CNOTs'''
    # store what is each qubit occupied by currently
    q_occupancy = [[]]*num_qubits
    for i in range(num_CNOT):
        q_c = q[int(CNOT_input[i, 0])]
        q_t = q[int(CNOT_input[i, 1])]
        o_d = []
        if q_occupancy[q_c[1]] != []:
            o_d.append(q_occupancy[q_c[1]])
        if q_occupancy[q_t[1]] != []:
            o_d.append(q_occupancy[q_t[1]])
        new_CNOT = OperationCNOT(q_c, q_t, o_d)
        total_CNOT.append(new_CNOT)
        # refresh q_occupancy
        q_occupancy[q_c[1]] = new_CNOT
        q_occupancy[q_t[1]] = new_CNOT
        
        if isinstance(cir, QuantumCircuit):
            new_CNOT.ConductOperation(cir)
    
    return total_CNOT

def CreateCNOTRandomlyOneLayer(q_log, num_CNOT, AG=None):
    '''
    generate CNOT operation randomly in only one layer
    it will not contain executable CNOT gates if AG is given
    input:
        q_reg: quantum register
        list of all operatuions
        if AG != None, the executable SWAPs will be removed
    output:
        list [(v_c, v_t), (v_c, v_t), ...]
        list [operation, ...]
    '''
    q = q_log
    num_qubits = len(q)
    pos = list(range(num_qubits))
    np.random.shuffle(pos)
    CNOT_operations = []
    CNOT_list = []
    flag_occupy = [0] * len(q_log)
    q_index_c = 0
    num_cx_current = 0
    edges = AG.edges if AG != None else [None]
    while num_cx_current < num_CNOT and q_index_c < num_qubits-1:
        q_index_t = q_index_c + 1
        q_c = pos[q_index_c]
        while q_index_t <= num_qubits-1:
            q_t = pos[q_index_t]
            if ((q_c, q_t) in edges or flag_occupy[q_c] == 1 or
                flag_occupy[q_t] == 1):
                # uneligible CNOT
                q_index_t += 1
            else:
                # eligible CNOT
                new_CNOT = OperationCNOT(q_c, q_t, [])
                CNOT_operations.append(new_CNOT)
                CNOT_list.append((q_c, q_t))
                num_cx_current += 1
                flag_occupy[q_c], flag_occupy[q_t] = 1, 1
                break
        q_index_c += 1
    return CNOT_list, CNOT_operations

def CreateCircuitFromQASM(file, path=None):
    #QASM_file = open('inputs/QASM example/' + file, 'r')
    if path == None:
        path = 'D:/pythonProject/profiling/circuit_slices_qft_14/'
    #QASM_file = open('D:/anaconda3/envs/quantum/Lib/site-packages/circuittransform/inputs/QASM example/100_cx_increase_q/' + file,
    #                 'r')
    QASM_file = open(path + file, 'r')
    iter_f = iter(QASM_file)
    QASM = ''
    for line in iter_f: #遍历文件，一行行遍历，读取文本
        QASM = QASM + line
    #print(QASM)
    cir = QuantumCircuit.from_qasm_str(QASM)
    QASM_file.close
    
    return cir

def CreateQASMFilesFromExample():
    path = './qasm/'
    files = os.listdir(path)
    
    return files

def GenerateEdgeofArchitectureGraph(vertex, method):
    edge = []
    num_vertex = len(vertex)
    if method[0] == 'circle' or method[0] == 'directed circle':        
        for i in range(num_vertex-1):
            edge.append((i, i+1))
        edge.append((num_vertex-1, 0))  
    
    '''
    grid architecturew with length = additional_arg[0], width = additional_arg[1]
    '''
    if method[0] == 'grid' or method[0] == 'directed grid':
        length = method[1]
        width = method[2]
        for raw in range(width-1):
            for col in range(length-1):
                current_v = col + raw*length
                edge.append((current_v, current_v + 1))
                edge.append((current_v, current_v + length))
        for raw in range(width-1):
            current_v = (length - 1) + raw*length
            edge.append((current_v, current_v + length))
        for col in range(length-1):
            current_v = col + (width - 1)*length
            edge.append((current_v, current_v + 1))
            
    if method[0] == 'grid2':
        length = method[1]
        width = method[2]
        for raw in range(width-1):
            for col in range(length-1):
                current_v = col + raw*length
                edge.append((current_v, current_v + 1))
                edge.append((current_v, current_v + length))
        for raw in range(width-1):
            current_v = (length - 1) + raw*length
            edge.append((current_v, current_v + length))
        for col in range(length-1):
            current_v = col + (width - 1)*length
            edge.append((current_v, current_v + 1))
        
        current_node1 = length*width
        for raw in [0, width-1]:
            for col in range(length):
                current_node2 = raw*length + col
                edge.append((current_node1, current_node2))
                current_node1 += 1
                
        for raw in [0, length-1]:
            for col in range(width):
                current_node2 = raw + col*length
                edge.append((current_node1, current_node2))
                current_node1 += 1
            
    return edge

def GenerateArchitectureGraph(method, draw_architecture_graph=False):
    '''
    generate architecture graph
    Input:
        method:
            IBM QX3
            IBM QX4
            IBM QX5
            IBM QX20
            IBM J-P
            IBM A-B-S
            IBM Rochester
            IBM Melbourne
            IBM Santiago
            Grid 6*6
            Grid 7*7
            Grid 8*8
            example in paper
    '''
    if method == ['IBM QX3']:
        G = GenerateArchitectureGraph(16, ['grid', 8, 2])
        G.remove_edges_from([(1, 9),(4, 5)])
        if draw_architecture_graph == True: nx.draw(G, with_labels=True)
        return G

    if method == ['IBM QX5']:
        G = nx.DiGraph()
        vertex = list(range(16))
        edges = [(1,2), (1,0), (2,3), (3,4), (3,14), (5,4), (6,5), (6,11), (6,7),\
                  (7,10), (8,7), (9,8), (9,10), (11,10), (12,5), (12,11), (12,13),\
                  (13,14), (13,4), (15,14), (15,2), (15,0)]
        G.add_nodes_from(vertex)
        G.add_edges_from(edges)
        if draw_architecture_graph == True: nx.draw(G, with_labels=True)
        return G

    if method == ['IBM QX20']:
        G = nx.Graph()
        num_q = 20
        vertex = list(range(20))
        edges = [(0,1),(1,2),(2,3),(3,4),(0,5),(1,6),(2,7),(3,8),(4,9),(1,7),(2,6),(3,9),(4,8),\
                 (5,6),(6,7),(7,8),(8,9),(5,10),(6,11),(7,12),(8,13),(9,14),(5,11),(6,10),(7,13),(8,12),\
                 (10,11),(11,12),(12,13),(13,14),(10,15),(11,16),(12,17),(13,18),(14,19),(11,17),(12,16),(13,19),(14,18),\
                 (15,16),(16,17),(17,18),(18,19)]
        G.add_nodes_from(vertex)
        G.add_edges_from(edges)
        if draw_architecture_graph == True: nx.draw(G, with_labels=True)
        return G, num_q

    if method == ['IBM J-P']:
        G = nx.Graph()
        vertex = list(range(20))
        num_q = 20
        edges = [(0,1),(1,2),(2,3),(3,4),(0,5),(4,9),
                 (5,6),(6,7),(7,8),(8,9),(5,10),(7,12),(9,14),
                 (10,11),(11,12),(12,13),(13,14),(10,15),(14,19),
                 (15,16),(16,17),(17,18),(18,19)]
        G.add_nodes_from(vertex)
        G.add_edges_from(edges)
        if draw_architecture_graph == True: nx.draw(G, with_labels=True)
        return G, num_q

    if method == ['IBM A-B-S']:
        G = nx.Graph()
        vertex = list(range(20))
        num_q = 20
        edges = [(0,1),(1,2),(2,3),(3,4),(1,6),(3,8),
                 (5,6),(6,7),(7,8),(8,9),(5,10),(7,12),(9,14),
                 (10,11),(11,12),(12,13),(13,14),(11,16),(13,18),
                 (15,16),(16,17),(17,18),(18,19)]
        G.add_nodes_from(vertex)
        G.add_edges_from(edges)
        if draw_architecture_graph == True: nx.draw(G, with_labels=True)
        return G, num_q
    
    if method == ['IBM Rochester']:
        AG = nx.Graph()
        vertex = list(range(53))
        num_q = 53
        edges = [(0,1),(1,2),(2,3),(3,4),(0,5),(4,6),(5,9),(6,13),
                 (7,8),(8,9),(9,10),(10,11),(11,12),(12,13),(13,14),(14,15),
                 (7,16),(11,17),(15,18),(16,19),(17,23),(18,27),
                 (19,20),(20,21),(21,22),(22,23),(23,24),(24,25),(25,26),(26,27),
                 (21,28),(25,29),(28,32),(29,36),
                 (30,31),(31,32),(32,33),(33,34),(34,35),(35,36),(36,37),(37,38),
                 (30,39),(34,40),(38,41),(39,42),(40,46),(41,50),
                 (42,43),(43,44),(44,45),(45,46),(46,47),(47,48),(48,49),(49,50),
                 (44,51),(48,52)]
        AG.add_nodes_from(vertex)
        AG.add_edges_from(edges)
        if draw_architecture_graph == True: nx.draw(AG, with_labels=True)
        return AG, num_q

    # 15 qubits melbourne
    # if method == ['IBM Melbourne']:
    #     G = nx.Graph()
    #     vertex = list(range(15))
    #     num_q = 15
    #     edges = [(0, 1), (0, 14), (1, 0), (1, 2), (1, 13), (2, 1),
    #              (2, 3), (2, 12), (3, 2), (3, 4), (3, 11),
    #              (4, 3), (4, 5), (4, 10), (5, 4), (5, 6), (5, 9),
    #              (6, 5), (6, 8), (7, 8), (8, 6), (8, 7), (8, 9),
    #              (9, 5), (9, 8), (9, 10), (10, 4), (10, 9), (10, 11),
    #              (11, 3), (11, 10), (11, 12), (12, 2),
    #              (12, 11), (12, 13), (13, 1), (13, 12), (13, 14), (14, 0), (14, 13)]
    #     G.add_nodes_from(vertex)
    #     G.add_edges_from(edges)
    #     if draw_architecture_graph == True: nx.draw(G, with_labels=True)
    #     return G, num_q
    # 14 qubits Melbourne
    if method == ['IBM Melbourne']:
        G = nx.Graph()
        vertex = list(range(14))
        num_q = 14
        edges = [(1, 0), (1, 2), (2, 3), (4, 3), (4, 10), (5, 4),
                (5, 6), (5, 9), (6, 8), (7, 8), (9, 8), (9, 10),
                (11, 3), (11, 10), (11, 12), (12, 2), (13, 1), (13, 12)]
        G.add_nodes_from(vertex)
        G.add_edges_from(edges)
        if draw_architecture_graph == True: nx.draw(G, with_labels=True)
        return G, num_q

    if method == ['IBM Santiago']:
        G = nx.Graph()
        vertex = list(range(5))
        num_q = 5
        edges = [[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2], [3, 4], [4, 3]]
        G.add_nodes_from(vertex)
        G.add_edges_from(edges)
        if draw_architecture_graph == True: nx.draw(G, with_labels=True)
        return G, num_q

    if method == ['IBM QX4']:
        G = nx.DiGraph()
        vertex = list(range(5))
        edges = [(1,0), (2,0), (2,1), (2,4), (3,4), (3,2)]
        G.add_nodes_from(vertex)
        G.add_edges_from(edges)
        if draw_architecture_graph == True: nx.draw(G, with_labels=True)
        return G
    
    if method == ['example in paper']:
        G = nx.DiGraph()
        vertex = list(range(6))
        edges = [(1,2), (1,0), (2,3), (3,4), (5,4), (5,2), (5,0)]
        G.add_nodes_from(vertex)
        G.add_edges_from(edges)
        if draw_architecture_graph == True: nx.draw(G, with_labels=True)
        return G        
    
    if method == ['Grid 8*8']:
        AG = nx.Graph()
        num_q = 64
        vertex = list(range(num_q))
        edges = GenerateEdgeofArchitectureGraph([], ['grid', 8, 8])
        AG.add_nodes_from(vertex)
        AG.add_edges_from(edges)
        if draw_architecture_graph == True: nx.draw(AG, with_labels=True)
        return AG, num_q

    if method == ['Grid 7*7']:
        AG = nx.Graph()
        num_q = 49
        vertex = list(range(num_q))
        edges = GenerateEdgeofArchitectureGraph([], ['grid', 7, 7])
        AG.add_nodes_from(vertex)
        AG.add_edges_from(edges)
        if draw_architecture_graph == True: nx.draw(AG, with_labels=True)
        return AG, num_q
    
    if method == ['Grid 6*6']:
        AG = nx.Graph()
        num_q = 36
        vertex = list(range(num_q))
        edges = GenerateEdgeofArchitectureGraph([], ['grid', 6, 6])
        AG.add_nodes_from(vertex)
        AG.add_edges_from(edges)
        if draw_architecture_graph == True: nx.draw(AG, with_labels=True)
        return AG, num_q
    
    if method == ['Grid 5*5']:
        AG = nx.Graph()
        num_q = 25
        vertex = list(range(num_q))
        edges = GenerateEdgeofArchitectureGraph([], ['grid', 5, 5])
        AG.add_nodes_from(vertex)
        AG.add_edges_from(edges)
        if draw_architecture_graph == True: nx.draw(AG, with_labels=True)
        return AG, num_q
    
    if method == ['Grid 5*4']:
        AG = nx.Graph()
        num_q = 20
        vertex = list(range(num_q))
        edges = GenerateEdgeofArchitectureGraph([], ['grid', 5, 4])
        AG.add_nodes_from(vertex)
        AG.add_edges_from(edges)
        if draw_architecture_graph == True: nx.draw(AG, with_labels=True)
        return AG, num_q
    
    if method == ['Grid 4*4']:
        AG = nx.Graph()
        num_q = 16
        vertex = list(range(num_q))
        edges = GenerateEdgeofArchitectureGraph([], ['grid', 4, 4])
        AG.add_nodes_from(vertex)
        AG.add_edges_from(edges)
        if draw_architecture_graph == True: nx.draw(AG, with_labels=True)
        return AG, num_q

    if method == ['Grid 2*3']:
        AG = nx.Graph()
        num_q = 6
        vertex = list(range(num_q))
        edges = GenerateEdgeofArchitectureGraph([], ['grid', 2, 3])
        AG.add_nodes_from(vertex)
        AG.add_edges_from(edges)
        if draw_architecture_graph == True: nx.draw(AG, with_labels=True)
        return AG, num_q
    
    raise(Exception('Unsupported AG %s' %method))
    
    return G

def GenerateArchitectureGraphError(num_vertex,
                                   method_AG,
                                   method_error,
                                   draw_architecture_graph = False):
    '''
    AG with dicts error_??? to represent error ??? in each node and edge
    '''
    AG = GenerateArchitectureGraph(num_vertex,
                                   method_AG,
                                   draw_architecture_graph)
    error_CX = {}
    error_single = {}
    if method_error[0] == 'test':
        for node in AG.nodes():
            error_single[node] = (np.random.rand()+0.5) * 0.01
        for edge in AG.edges():
            error_CX[node] = (np.random.rand()+0.5) * 0.001
        AG.error_single = error_single
        AG.error_CX = error_CX
    return AG

def CreatePartyMapRandomly(num_qubit, num_CNOT, q_reg):
    '''
    create party map randomly
    '''
    operation_CNOT = CreateCNOTRandomly(q_reg, num_CNOT)
    party_map = np.eye(num_qubit)
    for operation in operation_CNOT:
        c_raw = operation.control_qubit[1]
        t_raw = operation.target_qubit[1]
        party_map[t_raw][:] = np.logical_xor(party_map[t_raw][:], party_map[c_raw][:])
    
    '''set all diagonal elements to 1, it is for testing'''
    '''
    for i in range(num_qubit):
        party_map[i][i] = 1
    '''
    return party_map, operation_CNOT

def GenerateDependency(operations, num_q):
    '''Generate Dependency to operations according to the order'''
    dic = {}
    for i in range(num_q):
        dic[i] = None
    
    for operation in operations:
        qubits = operation.involve_qubits
        for q in qubits:
            if isinstance(q, int):
                q_index = q
            else:
                q_index = q.index
            if dic[q_index] == None:
                dic[q_index] = operation
            else:
                dependent_operation = dic[q_index]
                if not dependent_operation in operation.dependent_operations:
                    operation.dependent_operations.append(dependent_operation)
                dic[q_index] = operation

def CreateDGfromQASMfile(QASM_file, flag_single=False):
    '''
    convert QASM file to cir and DG
    flag_single: whether we should convert single qubit gate
    output:
        circuit, (DG, num_unidentified_gates, quantum register, operations)
    '''
    from Qiskitconverter.QiskitcircuittoDG import QiskitCircuitToDG
    cir = CreateCircuitFromQASM(QASM_file)
    res = QiskitCircuitToDG(cir, flag_single)
    return cir, res

def CreateControlFlow(mode, AG):
    '''
    mode = ['Harvard', num_gates_each_node]
    '''
    from circuittransform.inputs.controlflowgraph import ControlFlowGraph
    if mode[0] == 'Harvard':
        '''
        arXiv:1907.07113
        mode[1]: number of gates for each node in CFG, we will generate these
                 gates randomly, i.e., 50% CX and 50% single-qubit gates
        '''
        num_gates = mode[1]
        num_q_phy = len(AG.nodes())
        nodes = list(range(8))
        measure_qs = [num_q_phy-1, None, num_q_phy-1, None, num_q_phy-1,
                      num_q_phy-1, None, None]
        edges = [(0,1,0),(0,2,1),(1,0,None),(2,3,0),(2,4,1),(3,2,None),(4,5,0),
                 (4,6,1),(5,2,0),(5,7,1),(6,7,None)]
        '''ini CFG'''
        CFG = ControlFlowGraph(num_q_phy)
        CFG.AddAG(AG)
        '''add nodes'''
        for node, measure_q in zip(nodes, measure_qs):
            CFG.AddNode(node, measure_q)
            op_list = []
            op_cx_list = []
            '''gen gates to current node randomly and uniformly'''
            '''assume num_q_log = num_q_phy'''
            for _ in range(num_gates):
                q_lists = np.random.permutation(num_q_phy)
                if np.random.rand() < 0.5:
                    '''add CNOT'''
                    op_add = OperationCNOT(int(q_lists[0]), int(q_lists[1]))
                    op_cx_list.append(OperationCNOT(int(q_lists[0]), int(q_lists[1])))
                else:
                    '''use U3'''
                    paras = np.random.rand(3) * np.pi * 2
                    op_add = OperationU3(int(q_lists[0]), paras)
                    '''use Clifford'''
# =============================================================================
#                     gate = np.random.randint(3)
#                     if gate == 0:
#                         op_add = OperationSingle(int(q_lists[0]), 'h')
#                     if gate == 1:
#                         op_add = OperationSingle(int(q_lists[0]), 's')
#                     if gate == 2:
#                         op_add = OperationSingle(int(q_lists[0]), 'sdg')
# =============================================================================               
                op_list.append(op_add)
            '''store op list to current node'''
            CFG.AddOperationList(op_list, node)
            '''gen dependency to gates'''
            GenerateDependency(op_list, num_q_phy)
            GenerateDependency(op_cx_list, num_q_phy)
            '''gen and add DG'''
            DG = ct.OperationToDependencyGraph(op_list)
            DC_cx = ct.OperationToDependencyGraph(op_cx_list)
            CFG.AddDG(DG, node)
            CFG.nodes[node]['DG_cx'] = DC_cx
        '''add edges'''
        for edge in edges: CFG.AddEdge(edge[0:2], edge[2])
        return CFG
    
def CreateCNOTList(DG):
    CNOT_list = []
    DG_copy = copy.deepcopy(DG)
    leaf_nodes = ct.FindExecutableNode(DG_copy)
    while len(leaf_nodes) > 0:
        for node in leaf_nodes:
            op = DG_copy.nodes[node]['operation']
            add_CNOT = [op.involve_qubits[0].index, op.involve_qubits[1].index]
            CNOT_list.append(add_CNOT)
        DG_copy.remove_nodes_from(leaf_nodes)
        leaf_nodes = ct.FindExecutableNode(DG_copy)
    
    return CNOT_list

def GenQasmRandomly(num_file, num_qubit, num_cx, path, start_num=0):
    for i in range(num_file):
        full_path = path + str(start_num+i) + '.qasm'
        print('\rGenerating %d of %d file' %(i+1, num_file), end='')
        with open(full_path, 'w') as f:
            f.write('OPENQASM 2.0;\ninclude "qelib1.inc";')
            f.write('\nqreg q[' + str(num_qubit) + '];')
            f.write('\ncreg c[' + str(num_qubit) + '];')
            for _ in range(num_cx):
                cx1 = str(np.random.randint(num_qubit))
                cx2 = str(np.random.randint(num_qubit))
                while cx2 == cx1: cx2 = str(np.random.randint(num_qubit))
                f.write('\ncx q[' + cx1 + '],q[' + cx2 + '];')
        
if __name__ == '__main__':
    '''test GenerateArchitectureGraph'''
    '''
    l=4
    w=3
    num_vertex=l*w+2*(l+w)
    G = GenerateArchitectureGraph(num_vertex, ['grid2', l, w], draw_architecture_graph = True)
    '''
    
    '''test GenerateDependency'''
    '''
    num_q = 5
    num_CNOT = 5
    q = QuantumRegister(num_q, 'q')
    cir = QuantumCircuit(q)
    operations = CreateCNOTRandomly(q, num_CNOT, cir)
    GenerateDependency(operations, num_q)
    print(cir.draw())
    DG = ct.OperationToDependencyGraph(operations)
    nx.draw(DG, with_labels=True)
    '''
    
# =============================================================================
#     q = QuantumRegister(4, 'q')
#     cir = QuantumCircuit(q)
#     cir.cx(q[1], q[0])
#     cir.cx(q[1], q[2])
#     cir.cx(q[2], q[3])
#     cir.h(q[3])
#     res = ct.QiskitCircuitToDG(cir)
#     DG = res[0]
#     print(cir.draw())
#     nx.draw(DG, with_labels=True)
# =============================================================================
