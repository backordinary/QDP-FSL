# https://github.com/hflash/profiling/blob/0498dff8c1901591d4428c3149eb3aadf80ac483/circuittransform/main_benchmark_qasm.py
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 15:45:37 2019

@author: zxz58
"""

import circuittransform as ct
import networkx as nx
from networkx import DiGraph, Graph
import numpy as np
import os
from qiskit import QuantumCircuit, execute
from qiskit import QuantumRegister
from circuittransform import OperationU, OperationCNOT, OperationSWAP, Map
from circuittransform.map import InitialMapAStar
import matplotlib.pyplot as plt
import copy
import json
import time
#from inputs.get_benchmarks import GetBenchmarks, GetMappings

def get_initial_mapping(filedir):
    '''initialize parameters'''
    # choose quantum circuits
    # QASM_files = ct.CreateQASMFilesFromExample()
    # repeat time for simulated annealing
    repeat_time = 1
    # architecture graph generation control
    #method_AG = ['IBM QX3']
    #method_AG = ['IBM QX4']
    #method_AG = ['IBM QX5']
    #method_AG = ['IBM QX20']
    #method_AG = ['IBM J-P']
    #method_AG = ['IBM A-B-S']
    #method_AG = ['IBM Rochester']
    method_AG = ['IBM Melbourne']
    #method_AG = ['IBM Santiago']
    #method_AG = ['directed grid', 3, 3]
    #method_AG = ['Grid 6*6']
    #method_AG = ['Grid 4*4']
    '''initial mapping method'''
    initial_mapping_control = 3#0: naive; 3: annealing search; 4: specified by list;

    initial_map_list =  [9, 10, 8, 13, 6, 14, 5, 11, 4, 7, 12, 15, 1, 2, 17, 18, 0, 3, 16, 19]#only used for initial_mapping_control = 4
    '''method control'''
    use_RemotoCNOTandWindowLookAhead1 = 1

    '''generate architecture graph'''
    '''q0 - v0, q1 - v1, ...'''
    G, num_vertex = ct.GenerateArchitectureGraph(method_AG)
    num_qubits = num_vertex

    '''QASM input control'''
    QASM_files = os.listdir(filedir)

    print('QASM file is', QASM_files)
    '''output control'''
    out_num_swaps = False
    out_num_add_gates = True

    x_label = []
    x_lable_filename = []

    results = {}
    num_consider_gates = 0.5#counted gates for annealing search, 0-1 represents number gates * 0-1

    DiG = None
    if isinstance(G, DiGraph): #check whether it is a directed graph
        DiG = G
        G = nx.Graph(DiG)
    '''calculate shortest path and its length'''
    #shortest_path_G = nx.shortest_path(G, source=None, target=None, weight=None, method='dijkstra')
    #shortest_length_G = dict(nx.shortest_path_length(G, source=None, target=None, weight=None, method='dijkstra'))
    if DiG == None:
        res = ct.ShortestPath(G)
        shortest_path_G = res[1]
        shortest_length_G = (res[0], res[2])
    else:
        res = ct.ShortestPath(DiG)
        shortest_path_G = res[1]
        shortest_length_G = (res[0], res[2])

    '''use all possible swaps in parallel'''
    # =============================================================================
    # if imoprt_swaps_combination_from_json == True:
    #     fileObject = open('inputs\\swaps for architecture graph\\'+method_AG[0]+'.json', 'r')
    #     possible_swap_combination = json.load(fileObject)
    #     fileObject.close()
    # else:
    #     if use_Astar_search == True or use_Astar_lookahead == True or use_RemotoCNOTandWindow == True or use_UDecompositionFullConnectivity == True or use_HeuristicGreedySearch == True:
    #         possible_swap_combination = ct.FindAllPossibleSWAPParallel(G)
    # =============================================================================
    '''only use single swap'''
    possible_swap_combination = []
    edges = list(G.edges()).copy()
    for current_edge in edges:
        possible_swap_combination.append([current_edge])

    num_file = 0
    original_cir_size = []

    for file in QASM_files:
        if file[-5:] != '.qasm': file += '.qasm'
        num_file += 1
        res_qasm = ct.CreateDGfromQASMfile(file)
        x_lable_filename.append(file)
        results[file[0:-5]] = {}
        results[file[0:-5]]['initial map'] = []
        results[file[0:-5]]['initial map time'] = []
        results[file[0:-5]]['gates'] = []
        results[file[0:-5]]['gates time'] = []
        print('=============')
        print('Circuit name is', file)
        for repeat in range(repeat_time):
            print('------------')
            print('The repeated time is ', repeat)
            '''initialize logical quantum circuit'''
            q_log = res_qasm[1][2]
            cir_log = res_qasm[0]
            x_label.append(cir_log.size())
            if repeat == 0: original_cir_size.append(cir_log.size())

            '''initialize physical quantum circuit'''
            q_phy = QuantumRegister(num_vertex, 'v')
            cir_phy = QuantumCircuit(q_phy)

            '''generate architecture graph'''
            '''q0 - v0, q1 - v1, ...'''
            '''
            G = ct.GenerateArchitectureGraph(num_vertex, method_AG)
            '''

            '''calculate shortest path and its length'''
            '''
            shortest_path_G = nx.shortest_path(G, source=None, target=None, weight=None, method='dijkstra')
            shortest_length_G = dict(nx.shortest_path_length(G, source=None, target=None, weight=None, method='dijkstra'))
            if draw_architecture_graph == True: nx.draw(G, with_labels=True)
            '''

            '''generate CNOT operation'''
            total_CNOT = res_qasm[1][3]

            '''generate dependency graph'''
            DG = res_qasm[1][0]

            '''initialize map from logical qubits to physical qubits'''
            '''1-1, 2-2 ...'''
            if initial_mapping_control == 0:
                t_s = time.time()
                initial_map = Map(q_log, G)
                t_e = time.time()
            '''for circuit with only 10 qubit, we mannually map last 5 qubits to the down line'''
            if initial_mapping_control == 2:
                initial_map = Map(q_log, G)
                initial_map.RenewMapViaExchangeCod(9, 15)
                initial_map.RenewMapViaExchangeCod(8, 14)
                initial_map.RenewMapViaExchangeCod(7, 13)
                initial_map.RenewMapViaExchangeCod(6, 12)
            '''specific initial map through vertex list in AG'''
    # =============================================================================
    #         initial_map = Map(q_log, G, [1,2,3,8,7,6,11,12,13,16,17,18,4,9,14,19])
    # =============================================================================
            '''optimized initial mapping'''
            if initial_mapping_control == 1:
                t_s = time.time()
                map_res = ct.FindInitialMapping(DG, q_log, G, shortest_length_G[0])
                t_e = time.time()
                initial_map = map_res[0]
                print('initial_map is', map_res[1])
            '''annealing search'''
            if initial_mapping_control == 3:
                start_map = ct.FindInitialMapping(DG, q_log, G, shortest_length_G[0])
                t_s = time.time()
                map_res = ct.InitialMapSimulatedAnnealing(start_map[1], DG, G, DiG, q_log, shortest_length_G[0], shortest_path_G, num_consider_gates)
                t_e = time.time()
                initial_map = map_res[0]
                initial_map_list = map_res[1]
            if initial_mapping_control == 4:
                t_s = time.time()
                initial_map = Map(q_log, G, initial_map_list)
                t_e = time.time()

            results[file[0:-5]]['initial map'].append(initial_map_list)
            results[file[0:-5]]['initial map time'].append(t_e - t_s)

            if use_RemotoCNOTandWindowLookAhead1 == True:
                t_s = time.time()
                res = ct.RemoteCNOTandWindowLookAhead(q_phy, cir_phy, G,
                                                      copy.deepcopy(DG), initial_map,
                                                      shortest_length_G,
                                                      shortest_path_G,
                                                      depth_lookahead=1,
                                                      use_prune=True, draw=0,
                                                      DiG=DiG)
                t_e = time.time()
                print('Time consumption', t_e - t_s)
                print('Input gates number', cir_log.size())
                if out_num_add_gates == True: cost_RemotoCNOTandWindowLookAhead = res[3] + cir_log.size()
                print('Output gates number', cost_RemotoCNOTandWindowLookAhead)
                results[file[0:-5]]['gates'].append(cost_RemotoCNOTandWindowLookAhead)
                results[file[0:-5]]['gates time'].append(t_e - t_s)

    '''data processing'''
    post_res = []
    post_res_t1 = []#time for initial map
    post_res_t2 = []#time for searching
    post_res_map= []
    for name in QASM_files:#results.keys():
        if name[-5:] == '.qasm': name = name[0:-5]
        best_num = None#number of output gates
        best_t1 = None
        best_t2 = None
        best_map = None
        pos = -1
        for num_gate in results[name]['gates']:
            pos += 1
            if best_num == None:
                best_num = num_gate
                best_map = results[name]['initial map'][pos]
            else:
                if num_gate < best_num:
                    best_num = num_gate
                    best_map = results[name]['initial map'][pos]
        for num in results[name]['initial map time']:
            if best_t1 == None:
                best_t1 = num
            else:
                if num < best_t1:
                    best_t1 = num
        for num in results[name]['gates time']:
            if best_t2 == None:
                best_t2 = num
            else:
                if num < best_t2:
                    best_t2 = num
        post_res.append(best_num)
        post_res_t1.append(best_t1)
        post_res_t2.append(best_t2)
        post_res_map.append(best_map)
        return results

if __name__ == '__main__':
    filedir = 'D:/pythonProject/profiling/circuit_slices_qft_15/'
    results = get_initial_mapping(filedir)
    for item in results:
        print(item)
