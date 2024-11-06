# https://github.com/filipmazurek/Quantum-Workspace/blob/e2df6cfafa8664cd5fba8786ccf6e0540a0584fd/testing_tsp_and_maxcut.py
import tsp_qaoa_forest

import networkx as nx

from qiskit import Aer
from qiskit.tools.visualization import plot_histogram
from qiskit.aqua.input import EnergyInput
from qiskit.aqua.translators.ising import max_cut
from qiskit.aqua.algorithms import VQE
from qiskit.aqua.components.optimizers import SPSA
from qiskit.aqua.components.variational_forms import RY
from qiskit.aqua import QuantumInstance
from itertools import permutations


def run_pyquil_qaoa():
    """
    Problem with the whole program: Are all the matrices row by row always?
    :return:
    """

    cities = [[0, 0], [0, 1]]

    distance_matrix = get_distance_matrix(cities)

    print()
    print("Distance Matrix: ")
    print(distance_matrix)

    solver = tsp_qaoa_forest.ForestTSPSolver(distance_matrix, steps=5, use_constraints=True)

    solution, distribution = solver.solve_tsp()

    print()
    print("Solution: ")
    print(solution)

    print()
    print("Distribution: ")
    print(distribution)


def qiskit_maxcut():
    n = 4  # Number of nodes in graph
    G = nx.Graph()
    G.add_nodes_from(np.arange(0, n, 1))
    elist = [(0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0), (1, 2, 1.0), (2, 3, 1.0)]
    # tuple is (i,j,weight) where (i,j) is the edge
    G.add_weighted_edges_from(elist)

    colors = ['r' for node in G.nodes()]
    pos = nx.spring_layout(G)
    default_axes = plt.axes(frameon=True)
    nx.draw_networkx(G, node_color=colors, node_size=600, alpha=.8, ax=default_axes, pos=pos)
    w = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            temp = G.get_edge_data(i, j, default=0)
            if temp != 0:
                w[i, j] = temp['weight']
    # print(w)  # weight matrix. i.e. distances. 0 indicates no path.

    #####
    # Brute force solution
    # best_cost_brute = 0
    # for b in range(2 ** n):
    #     x = [int(t) for t in reversed(list(bin(b)[2:].zfill(n)))]
    #     cost = 0
    #     for i in range(n):
    #         for j in range(n):
    #             cost = cost + w[i, j] * x[i] * (1 - x[j])
    #     if best_cost_brute < cost:
    #         best_cost_brute = cost
    #         xbest_brute = x
    #     print('case = ' + str(x) + ' cost = ' + str(cost))
    #
    # colors = ['r' if xbest_brute[i] == 0 else 'b' for i in range(n)]
    # nx.draw_networkx(G, node_color=colors, node_size=600, alpha=.8, pos=pos)
    # print('\nBest solution = ' + str(xbest_brute) + ' cost = ' + str(best_cost_brute))
    # End brute force solution
    #####

    #####
    # Eigensolver solution
    # qubitOp, offset = max_cut.get_max_cut_qubitops(w)
    # algo_input = EnergyInput(qubitOp)
    #
    # # Making the Hamiltonian in its full form and getting the lowest eigenvalue and eigenvector
    # ee = ExactEigensolver(qubitOp, k=1)
    # result = ee.run()
    #
    # """
    # algorithm_cfg = {
    #     'name': 'ExactEigensolver',
    # }
    #
    # params = {
    #     'problem': {'name': 'ising'},
    #     'algorithm': algorithm_cfg
    # }
    # result = run_algorithm(params,algo_input)
    # """
    # x = max_cut.sample_most_likely(result['eigvecs'][0])
    # print('energy:', result['energy'])
    # print('maxcut objective:', result['energy'] + offset)
    # print('solution:', max_cut.get_graph_solution(x))
    # print('solution objective:', max_cut.max_cut_value(x, w))
    #
    # colors = ['r' if max_cut.get_graph_solution(x)[i] == 0 else 'b' for i in range(n)]
    # nx.draw_networkx(G, node_color=colors, node_size=600, alpha=.8, pos=pos)
    # End eigensolver solution
    #####

    # run quantum algorithm with shots
    qubitOp, offset = max_cut.get_max_cut_qubitops(w)
    algo_input = EnergyInput(qubitOp)
    seed = 10598

    spsa = SPSA(max_trials=300)
    ry = RY(qubitOp.num_qubits, depth=5, entanglement='linear')
    vqe = VQE(qubitOp, ry, spsa, 'grouped_paulis')

    backend = Aer.get_backend('qasm_simulator')
    quantum_instance = QuantumInstance(backend=backend, shots=1024, seed=seed)

    result = vqe.run(quantum_instance)

    """declarative approach, update the param from the previous cell.
    params['algorithm']['operator_mode'] = 'grouped_paulis'
    params['backend']['name'] = 'qasm_simulator'
    params['backend']['shots'] = 1024
    result = run_algorithm(params, algo_input)
    """

    x = max_cut.sample_most_likely(result['eigvecs'][0])
    print('energy:', result['energy'])
    print('time:', result['eval_time'])
    print('maxcut objective:', result['energy'] + offset)
    print('solution:', max_cut.get_graph_solution(x))
    print('solution objective:', max_cut.max_cut_value(x, w))
    plot_histogram(result['eigvecs'][0])

    colors = ['r' if max_cut.get_graph_solution(x)[i] == 0 else 'b' for i in range(n)]
    nx.draw_networkx(G, node_color=colors, node_size=600, alpha=.8, pos=pos)


def brute_force_tsp(w, N):
    a = list(permutations(range(1, N)))
    last_best_distance = 1e10
    for i in a:
        distance = 0
        pre_j = 0
        for j in i:
            distance = distance + w[j, pre_j]
            pre_j = j
        distance = distance + w[pre_j, 0]
        order = (0,) + i
        if distance < last_best_distance:
            best_order = order
            last_best_distance = distance
            print('order = ' + str(order) + ' Distance = ' + str(distance))
    return last_best_distance, best_order


def draw_tsp_solution(G, order, colors, pos):
    G2 = G.copy()
    n = len(order)
    for i in range(n):
        j = (i + 1) % n
        G2.add_edge(order[i], order[j])
    # default_axes = plt.axes(frameon=True)
    # nx.draw_networkx(G2, node_color=colors, node_size=600, alpha=.8, ax=default_axes, pos=pos)
    nx.draw_networkx(G2)


def qiskit_tsp():
    # Generating a graph of 2 nodes
    n = 2
    num_qubits = n ** 2
    ins = tsp.random_tsp(n)
    G = nx.Graph()
    G.add_nodes_from(np.arange(0, n, 1))
    colors = ['r' for node in G.nodes()]
    pos = {k: v for k, v in enumerate(ins.coord)}
    default_axes = plt.axes(frameon=True)
    # nx.draw_networkx(G, node_color=colors, node_size=600, alpha=.8, ax=default_axes, pos=pos)
    print('distance\n', ins.w)

    best_distance, best_order = brute_force_tsp(ins.w, ins.dim)
    print('Best order from brute force = ' + str(best_order) + ' with total distance = ' + str(best_distance))

    draw_tsp_solution(G, best_order, colors, pos)

    print()
    ########
    # Starting quantum algorithm
    ########
    qubitOp, offset = tsp.get_tsp_qubitops(ins)
    algo_input = EnergyInput(qubitOp)
    # Making the Hamiltonian in its full form and getting the lowest eigenvalue and eigenvector
    # ee = ExactEigensolver(qubitOp, k=1)
    # result = ee.run()
    #
    # """
    # algorithm_cfg = {
    #     'name': 'ExactEigensolver',
    # }
    #
    # params = {
    #     'problem': {'name': 'ising'},
    #     'algorithm': algorithm_cfg
    # }
    # result = run_algorithm(params,algo_input)
    # """
    # print('energy:', result['energy'])
    # # print('tsp objective:', result['energy'] + offset)
    # x = tsp.sample_most_likely(result['eigvecs'][0])
    # print('feasible:', tsp.tsp_feasible(x))
    # z = tsp.get_tsp_solution(x)
    # print('solution:', z)
    # print('solution objective:', tsp.tsp_value(z, ins.w))
    # draw_tsp_solution(G, z, colors, pos)
    #
    # seed = 10598
    #
    # spsa = SPSA(max_trials=300)
    # ry = RY(qubitOp.num_qubits, depth=5, entanglement='linear')
    # vqe = VQE(qubitOp, ry, spsa, 'matrix')
    #
    # backend = Aer.get_backend('statevector_simulator')
    # quantum_instance = QuantumInstance(backend=backend, seed=seed)
    #
    # result = vqe.run(quantum_instance)
    # """
    # algorithm_cfg = {
    #     'name': 'VQE',
    #     'operator_mode': 'matrix'
    # }
    #
    # optimizer_cfg = {
    #     'name': 'SPSA',
    #     'max_trials': 300
    # }
    #
    # var_form_cfg = {
    #     'name': 'RY',
    #     'depth': 5,
    #     'entanglement': 'linear'
    # }
    #
    # params = {
    #     'problem': {'name': 'ising', 'random_seed': seed},
    #     'algorithm': algorithm_cfg,
    #     'optimizer': optimizer_cfg,
    #     'variational_form': var_form_cfg,
    #     'backend': {'name': 'statevector_simulator'}
    # }
    # result = run_algorithm(parahms,algo_input)
    # """
    # print('energy:', result['energy'])
    # print('time:', result['eval_time'])
    # # print('tsp objective:', result['energy'] + offset)
    # x = tsp.sample_most_likely(result['eigvecs'][0])
    # print('feasible:', tsp.tsp_feasible(x))
    # z = tsp.get_tsp_solution(x)
    # print('solution:', z)
    # print('solution objective:', tsp.tsp_value(z, ins.w))
    # draw_tsp_solution(G, z, colors, pos)
    #

    ######
    # Quantum algorithm with shots
    ####

    seed = 10598

    spsa = SPSA(max_trials=300)
    ry = RY(qubitOp.num_qubits, depth=5, entanglement='linear')
    vqe = VQE(qubitOp, ry, spsa, 'grouped_paulis')

    backend = Aer.get_backend('qasm_simulator')
    quantum_instance = QuantumInstance(backend=backend, shots=1024, seed=seed)

    result = vqe.run(quantum_instance)

    """update params in the previous cell
    params['algorithm']['operator_mode'] = 'grouped_paulis'
    params['backend']['name'] = 'qasm_simulator'
    params['backend']['shots'] = 1024
    result = run_algorithm(params,algo_input)
    """
    print('energy:', result['energy'])
    print('time:', result['eval_time'])
    # print('tsp objective:', result['energy'] + offset)
    x = tsp.sample_most_likely(result['eigvecs'][0])
    print('feasible:', tsp.tsp_feasible(x))
    z = tsp.get_tsp_solution(x)
    print('solution:', z)
    print('solution objective:', tsp.tsp_value(z, ins.w))
    plot_histogram(result['eigvecs'][0])
    draw_tsp_solution(G, z, colors, pos)


from .rigetti_result_analysis import *
from pyquil.paulis import PauliTerm, PauliSum
from .forest_utils_ms import get_distance_matrix
from qiskit.aqua.translators.ising import tsp


def create_penalty_operators_for_qubit_range(range_of_qubits, distance_matrix):
    cost_operators = []
    weight = -100 * np.max(distance_matrix)
    for i in range_of_qubits:
        if i == range_of_qubits[0]:
            z_term = PauliTerm("Z", i, weight)
            all_ones_term = PauliTerm("I", 0, 0.5 * weight) - PauliTerm("Z", i, 0.5 * weight)
        else:
            z_term = z_term * PauliTerm("Z", i)
            all_ones_term = all_ones_term * (PauliTerm("I", 0, 0.5) - PauliTerm("Z", i, 0.5))

    z_term = PauliSum([z_term])

    cost_operators.append(PauliTerm("I", 0, weight) - z_term - all_ones_term)

    return cost_operators


"""
Please make sure to run the desired function down here. Otherwise nothing will run.
"""

run_pyquil_qaoa()
