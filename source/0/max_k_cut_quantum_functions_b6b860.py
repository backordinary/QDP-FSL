# https://github.com/fjelljenta/Smart-Charging/blob/f4f7a2fe26802690460dd5d20edb412ef613640a/web_app/console_simulation/Max_k_cut_quantum_functions.py
from console_simulation.data_processing import *
from qiskit import QuantumCircuit
from qiskit import Aer
from qiskit.compiler import transpile
from qiskit.circuit import Parameter
from scipy.optimize import differential_evolution, minimize
import numpy as np
from console_simulation.data_processing import *
from itertools import combinations
import numpy as np
import copy
import ast


# Classical functions


def powerset(l):
    """
    :param l: a list containing numbers usually
    :return: the powerset, or to say a set containing all the subsets, of the set of the list, in list format
    """
    res = []
    n = len(l)
    for i in range(0, n + 1):
        for element in combinations(l, i):
            res.append(element)
    return res


def compute_cost(counts, l, w, n_counts=512):
    """
    :param counts:  dict{measurement result in string:count}
    :param l: The number of qubits representing a node
    :param w: The weight matrix for edges
    :param n_counts: The total number of counts
    :return:  The averaged cost
    """
    total_cost = 0
    for measurement, count in counts.items():
        partition = [int(measurement[i:i + l], 2) for i in range(0, len(measurement), l)]
        for i in range(len(partition)):
            for j in range(i, len(partition)):
                if partition[i] != partition[j]:
                    total_cost += w[i][j] * count
    average_cost = total_cost / n_counts
    return average_cost


def show_distribution(counts, l):
    res = {}
    for measurement, count in counts.items():
        partition = [int(measurement[i:i + l], 2) for i in range(0, len(measurement), l)]
        partition_corrected = []
        group = []
        added = []
        for i in range(len(partition)):
            if i not in added:
                group.append(i)
                added.append(i)
                for j in range(i, len(partition)):
                    if partition[i] == partition[j] and j not in added:
                        group.append(j)
                        added.append(j)
            if len(group) != 0:
                partition_corrected.append(group)
            group = []
        added = []
        if str(partition_corrected) not in res.keys():
            res[str(partition_corrected)] = count
        else:
            res[str(partition_corrected)] += count
    return res


def Monte_Carlo_solver(G, k):
    """Classical Monte-carlo solution of the max-k-cut problem of Graph G
    Args:
        G (graph): Graph of the max-k-cut problem
        k (int): Number of cuts (available loading stations)
    Returns:
        C_opt (int): cost function of optimal max-k-cut solution
        P_opt (dict): dictionary with the number of the load station as keys
            and array of jobs at a given loading station (still unordered!) as values
    Method:
    """
    N = G.number_of_nodes()

    # Initialisation:

    P_opt = dict()
    for i in range(k):
        P_opt['P' + str(i)] = []

    for j in range(N):
        P_opt['P' + str(np.random.randint(k))].append(j)  # maybe change to actual nodes the graph

    C_opt = cost(G, P_opt)
    f = 0

    # Monte-Carlo:
    while f < 100:

        P = copy.deepcopy(P_opt)
        P_set_index_t = np.random.randint(k)
        set_length = len(P['P' + str(P_set_index_t)])
        if set_length == 0:
            continue
        Np_index = np.random.randint(set_length)
        MC_node = P['P' + str(P_set_index_t)].pop(Np_index)
        P_set_index_g = np.random.randint(k - 1)
        if P_set_index_g == P_set_index_t:
            P_set_index_g = k - 1
        P['P' + str(P_set_index_g)].append(MC_node)

        C = cost(G, P)

        probability = np.random.random()
        normalisation = 10.0

        if np.exp((C - C_opt) / normalisation) > probability:
            f = 0
            P_opt = copy.deepcopy(P)
            C_opt = C.copy()
            print('C_opt=', C_opt, '  P_opt=', P_opt, '\n')
        else:
            f += 1

    return C_opt, P_opt


def brut_force(G, k):
    """Classical brut-force solution of the max-k-cut problem of Graph G
    Args:
        G (graph): Graph of the max-k-cut problem
        k (int): Number of cuts (available loading stations)
    Globals:
        time (int): Used for progress output
    Returns:
        C_opt (int): cost function of optimal max-k-cut solution
        P_opt (dict): dictionary with the number of the load station as keys
            and array of jobs at a given loading station (still unordered!) as values
    Method:
        compute every posible k-cut, save the corresponding cost function
        and find the maximal index of the maximal cost
    """
    C_opt = 0
    P_opt = {'P0': [0]}
    for i in range(1, k):
        P_opt['P' + str(i)] = []

    N = G.number_of_nodes()
    N_rec = N - 1

    global time
    time = 0

    C_opt, P_opt = rec_cost_optimization(G, k, N, N_rec, P_opt, C_opt)
    print('{}\r'.format('progress ' + str(100) + '%' + '   '), end="")

    return C_opt, P_opt


def brut_force_k4_N5(G):
    """Classical brut-force solution of the max-4-cut problem of a Graph with 5 nodes
    Args:
        G (graph): Graph with 5 nodes of the max-4-cut problem
    Returns:
        C_opt (int): cost function of optimal max-4-cut solution
        P_opt (dict): dictionary with the number of the load station as keys
            and array of jobs at a given loading station (still unordered!) as values
    Method:
        compute every posible 4-cut, save the corresponding cost function
        and find the maximal index of the maximal cost
    """

    # TODO: at the moment only works if node names correspond to the number (int) of the node
    # jobs=list(G.nodes())

    P_opt = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    C_opt = 0
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(0)].append(1)
    P_try['P' + str(0)].append(2)
    P_try['P' + str(0)].append(3)
    P_try['P' + str(0)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(0)].append(1)
    P_try['P' + str(0)].append(2)
    P_try['P' + str(0)].append(3)
    P_try['P' + str(1)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(0)].append(1)
    P_try['P' + str(0)].append(2)
    P_try['P' + str(0)].append(3)
    P_try['P' + str(2)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(0)].append(1)
    P_try['P' + str(0)].append(2)
    P_try['P' + str(0)].append(3)
    P_try['P' + str(3)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(0)].append(1)
    P_try['P' + str(0)].append(2)
    P_try['P' + str(1)].append(3)
    P_try['P' + str(0)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(0)].append(1)
    P_try['P' + str(0)].append(2)
    P_try['P' + str(1)].append(3)
    P_try['P' + str(1)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(0)].append(1)
    P_try['P' + str(0)].append(2)
    P_try['P' + str(1)].append(3)
    P_try['P' + str(2)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(0)].append(1)
    P_try['P' + str(0)].append(2)
    P_try['P' + str(1)].append(3)
    P_try['P' + str(3)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(0)].append(1)
    P_try['P' + str(0)].append(2)
    P_try['P' + str(2)].append(3)
    P_try['P' + str(0)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(0)].append(1)
    P_try['P' + str(0)].append(2)
    P_try['P' + str(2)].append(3)
    P_try['P' + str(1)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(0)].append(1)
    P_try['P' + str(0)].append(2)
    P_try['P' + str(2)].append(3)
    P_try['P' + str(2)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(0)].append(1)
    P_try['P' + str(0)].append(2)
    P_try['P' + str(2)].append(3)
    P_try['P' + str(3)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(0)].append(1)
    P_try['P' + str(0)].append(2)
    P_try['P' + str(3)].append(3)
    P_try['P' + str(0)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(0)].append(1)
    P_try['P' + str(0)].append(2)
    P_try['P' + str(3)].append(3)
    P_try['P' + str(1)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(0)].append(1)
    P_try['P' + str(0)].append(2)
    P_try['P' + str(3)].append(3)
    P_try['P' + str(2)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(0)].append(1)
    P_try['P' + str(0)].append(2)
    P_try['P' + str(3)].append(3)
    P_try['P' + str(3)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(0)].append(1)
    P_try['P' + str(1)].append(2)
    P_try['P' + str(0)].append(3)
    P_try['P' + str(0)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(0)].append(1)
    P_try['P' + str(1)].append(2)
    P_try['P' + str(0)].append(3)
    P_try['P' + str(1)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(0)].append(1)
    P_try['P' + str(1)].append(2)
    P_try['P' + str(0)].append(3)
    P_try['P' + str(2)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(0)].append(1)
    P_try['P' + str(1)].append(2)
    P_try['P' + str(0)].append(3)
    P_try['P' + str(3)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(0)].append(1)
    P_try['P' + str(1)].append(2)
    P_try['P' + str(1)].append(3)
    P_try['P' + str(0)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(0)].append(1)
    P_try['P' + str(1)].append(2)
    P_try['P' + str(1)].append(3)
    P_try['P' + str(1)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(0)].append(1)
    P_try['P' + str(1)].append(2)
    P_try['P' + str(1)].append(3)
    P_try['P' + str(2)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(0)].append(1)
    P_try['P' + str(1)].append(2)
    P_try['P' + str(1)].append(3)
    P_try['P' + str(3)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(0)].append(1)
    P_try['P' + str(1)].append(2)
    P_try['P' + str(2)].append(3)
    P_try['P' + str(0)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(0)].append(1)
    P_try['P' + str(1)].append(2)
    P_try['P' + str(2)].append(3)
    P_try['P' + str(1)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(0)].append(1)
    P_try['P' + str(1)].append(2)
    P_try['P' + str(2)].append(3)
    P_try['P' + str(2)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(0)].append(1)
    P_try['P' + str(1)].append(2)
    P_try['P' + str(2)].append(3)
    P_try['P' + str(3)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(0)].append(1)
    P_try['P' + str(1)].append(2)
    P_try['P' + str(3)].append(3)
    P_try['P' + str(0)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(0)].append(1)
    P_try['P' + str(1)].append(2)
    P_try['P' + str(3)].append(3)
    P_try['P' + str(1)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(0)].append(1)
    P_try['P' + str(1)].append(2)
    P_try['P' + str(3)].append(3)
    P_try['P' + str(2)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(0)].append(1)
    P_try['P' + str(1)].append(2)
    P_try['P' + str(3)].append(3)
    P_try['P' + str(3)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(0)].append(1)
    P_try['P' + str(2)].append(2)
    P_try['P' + str(0)].append(3)
    P_try['P' + str(0)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(0)].append(1)
    P_try['P' + str(2)].append(2)
    P_try['P' + str(0)].append(3)
    P_try['P' + str(1)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(0)].append(1)
    P_try['P' + str(2)].append(2)
    P_try['P' + str(0)].append(3)
    P_try['P' + str(2)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(0)].append(1)
    P_try['P' + str(2)].append(2)
    P_try['P' + str(0)].append(3)
    P_try['P' + str(3)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(0)].append(1)
    P_try['P' + str(2)].append(2)
    P_try['P' + str(1)].append(3)
    P_try['P' + str(0)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(0)].append(1)
    P_try['P' + str(2)].append(2)
    P_try['P' + str(1)].append(3)
    P_try['P' + str(1)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(0)].append(1)
    P_try['P' + str(2)].append(2)
    P_try['P' + str(1)].append(3)
    P_try['P' + str(2)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(0)].append(1)
    P_try['P' + str(2)].append(2)
    P_try['P' + str(1)].append(3)
    P_try['P' + str(3)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(0)].append(1)
    P_try['P' + str(2)].append(2)
    P_try['P' + str(2)].append(3)
    P_try['P' + str(0)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(0)].append(1)
    P_try['P' + str(2)].append(2)
    P_try['P' + str(2)].append(3)
    P_try['P' + str(1)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(0)].append(1)
    P_try['P' + str(2)].append(2)
    P_try['P' + str(2)].append(3)
    P_try['P' + str(2)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(0)].append(1)
    P_try['P' + str(2)].append(2)
    P_try['P' + str(2)].append(3)
    P_try['P' + str(3)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(0)].append(1)
    P_try['P' + str(2)].append(2)
    P_try['P' + str(3)].append(3)
    P_try['P' + str(0)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(0)].append(1)
    P_try['P' + str(2)].append(2)
    P_try['P' + str(3)].append(3)
    P_try['P' + str(1)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(0)].append(1)
    P_try['P' + str(2)].append(2)
    P_try['P' + str(3)].append(3)
    P_try['P' + str(2)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(0)].append(1)
    P_try['P' + str(2)].append(2)
    P_try['P' + str(3)].append(3)
    P_try['P' + str(3)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(0)].append(1)
    P_try['P' + str(3)].append(2)
    P_try['P' + str(0)].append(3)
    P_try['P' + str(0)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(0)].append(1)
    P_try['P' + str(3)].append(2)
    P_try['P' + str(0)].append(3)
    P_try['P' + str(1)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(0)].append(1)
    P_try['P' + str(3)].append(2)
    P_try['P' + str(0)].append(3)
    P_try['P' + str(2)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(0)].append(1)
    P_try['P' + str(3)].append(2)
    P_try['P' + str(0)].append(3)
    P_try['P' + str(3)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(0)].append(1)
    P_try['P' + str(3)].append(2)
    P_try['P' + str(1)].append(3)
    P_try['P' + str(0)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(0)].append(1)
    P_try['P' + str(3)].append(2)
    P_try['P' + str(1)].append(3)
    P_try['P' + str(1)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(0)].append(1)
    P_try['P' + str(3)].append(2)
    P_try['P' + str(1)].append(3)
    P_try['P' + str(2)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(0)].append(1)
    P_try['P' + str(3)].append(2)
    P_try['P' + str(1)].append(3)
    P_try['P' + str(3)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(0)].append(1)
    P_try['P' + str(3)].append(2)
    P_try['P' + str(2)].append(3)
    P_try['P' + str(0)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(0)].append(1)
    P_try['P' + str(3)].append(2)
    P_try['P' + str(2)].append(3)
    P_try['P' + str(1)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(0)].append(1)
    P_try['P' + str(3)].append(2)
    P_try['P' + str(2)].append(3)
    P_try['P' + str(2)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(0)].append(1)
    P_try['P' + str(3)].append(2)
    P_try['P' + str(2)].append(3)
    P_try['P' + str(3)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(0)].append(1)
    P_try['P' + str(3)].append(2)
    P_try['P' + str(3)].append(3)
    P_try['P' + str(0)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(0)].append(1)
    P_try['P' + str(3)].append(2)
    P_try['P' + str(3)].append(3)
    P_try['P' + str(1)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(0)].append(1)
    P_try['P' + str(3)].append(2)
    P_try['P' + str(3)].append(3)
    P_try['P' + str(2)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(0)].append(1)
    P_try['P' + str(3)].append(2)
    P_try['P' + str(3)].append(3)
    P_try['P' + str(3)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(1)].append(1)
    P_try['P' + str(0)].append(2)
    P_try['P' + str(0)].append(3)
    P_try['P' + str(0)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(1)].append(1)
    P_try['P' + str(0)].append(2)
    P_try['P' + str(0)].append(3)
    P_try['P' + str(1)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(1)].append(1)
    P_try['P' + str(0)].append(2)
    P_try['P' + str(0)].append(3)
    P_try['P' + str(2)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(1)].append(1)
    P_try['P' + str(0)].append(2)
    P_try['P' + str(0)].append(3)
    P_try['P' + str(3)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(1)].append(1)
    P_try['P' + str(0)].append(2)
    P_try['P' + str(1)].append(3)
    P_try['P' + str(0)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(1)].append(1)
    P_try['P' + str(0)].append(2)
    P_try['P' + str(1)].append(3)
    P_try['P' + str(1)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(1)].append(1)
    P_try['P' + str(0)].append(2)
    P_try['P' + str(1)].append(3)
    P_try['P' + str(2)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(1)].append(1)
    P_try['P' + str(0)].append(2)
    P_try['P' + str(1)].append(3)
    P_try['P' + str(3)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(1)].append(1)
    P_try['P' + str(0)].append(2)
    P_try['P' + str(2)].append(3)
    P_try['P' + str(0)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(1)].append(1)
    P_try['P' + str(0)].append(2)
    P_try['P' + str(2)].append(3)
    P_try['P' + str(1)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(1)].append(1)
    P_try['P' + str(0)].append(2)
    P_try['P' + str(2)].append(3)
    P_try['P' + str(2)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(1)].append(1)
    P_try['P' + str(0)].append(2)
    P_try['P' + str(2)].append(3)
    P_try['P' + str(3)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(1)].append(1)
    P_try['P' + str(0)].append(2)
    P_try['P' + str(3)].append(3)
    P_try['P' + str(0)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(1)].append(1)
    P_try['P' + str(0)].append(2)
    P_try['P' + str(3)].append(3)
    P_try['P' + str(1)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(1)].append(1)
    P_try['P' + str(0)].append(2)
    P_try['P' + str(3)].append(3)
    P_try['P' + str(2)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(1)].append(1)
    P_try['P' + str(0)].append(2)
    P_try['P' + str(3)].append(3)
    P_try['P' + str(3)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(1)].append(1)
    P_try['P' + str(1)].append(2)
    P_try['P' + str(0)].append(3)
    P_try['P' + str(0)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(1)].append(1)
    P_try['P' + str(1)].append(2)
    P_try['P' + str(0)].append(3)
    P_try['P' + str(1)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(1)].append(1)
    P_try['P' + str(1)].append(2)
    P_try['P' + str(0)].append(3)
    P_try['P' + str(2)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(1)].append(1)
    P_try['P' + str(1)].append(2)
    P_try['P' + str(0)].append(3)
    P_try['P' + str(3)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(1)].append(1)
    P_try['P' + str(1)].append(2)
    P_try['P' + str(1)].append(3)
    P_try['P' + str(0)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(1)].append(1)
    P_try['P' + str(1)].append(2)
    P_try['P' + str(1)].append(3)
    P_try['P' + str(1)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(1)].append(1)
    P_try['P' + str(1)].append(2)
    P_try['P' + str(1)].append(3)
    P_try['P' + str(2)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(1)].append(1)
    P_try['P' + str(1)].append(2)
    P_try['P' + str(1)].append(3)
    P_try['P' + str(3)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(1)].append(1)
    P_try['P' + str(1)].append(2)
    P_try['P' + str(2)].append(3)
    P_try['P' + str(0)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(1)].append(1)
    P_try['P' + str(1)].append(2)
    P_try['P' + str(2)].append(3)
    P_try['P' + str(1)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(1)].append(1)
    P_try['P' + str(1)].append(2)
    P_try['P' + str(2)].append(3)
    P_try['P' + str(2)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(1)].append(1)
    P_try['P' + str(1)].append(2)
    P_try['P' + str(2)].append(3)
    P_try['P' + str(3)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(1)].append(1)
    P_try['P' + str(1)].append(2)
    P_try['P' + str(3)].append(3)
    P_try['P' + str(0)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(1)].append(1)
    P_try['P' + str(1)].append(2)
    P_try['P' + str(3)].append(3)
    P_try['P' + str(1)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(1)].append(1)
    P_try['P' + str(1)].append(2)
    P_try['P' + str(3)].append(3)
    P_try['P' + str(2)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(1)].append(1)
    P_try['P' + str(1)].append(2)
    P_try['P' + str(3)].append(3)
    P_try['P' + str(3)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(1)].append(1)
    P_try['P' + str(2)].append(2)
    P_try['P' + str(0)].append(3)
    P_try['P' + str(0)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(1)].append(1)
    P_try['P' + str(2)].append(2)
    P_try['P' + str(0)].append(3)
    P_try['P' + str(1)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(1)].append(1)
    P_try['P' + str(2)].append(2)
    P_try['P' + str(0)].append(3)
    P_try['P' + str(2)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(1)].append(1)
    P_try['P' + str(2)].append(2)
    P_try['P' + str(0)].append(3)
    P_try['P' + str(3)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(1)].append(1)
    P_try['P' + str(2)].append(2)
    P_try['P' + str(1)].append(3)
    P_try['P' + str(0)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(1)].append(1)
    P_try['P' + str(2)].append(2)
    P_try['P' + str(1)].append(3)
    P_try['P' + str(1)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(1)].append(1)
    P_try['P' + str(2)].append(2)
    P_try['P' + str(1)].append(3)
    P_try['P' + str(2)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(1)].append(1)
    P_try['P' + str(2)].append(2)
    P_try['P' + str(1)].append(3)
    P_try['P' + str(3)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(1)].append(1)
    P_try['P' + str(2)].append(2)
    P_try['P' + str(2)].append(3)
    P_try['P' + str(0)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(1)].append(1)
    P_try['P' + str(2)].append(2)
    P_try['P' + str(2)].append(3)
    P_try['P' + str(1)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(1)].append(1)
    P_try['P' + str(2)].append(2)
    P_try['P' + str(2)].append(3)
    P_try['P' + str(2)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(1)].append(1)
    P_try['P' + str(2)].append(2)
    P_try['P' + str(2)].append(3)
    P_try['P' + str(3)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(1)].append(1)
    P_try['P' + str(2)].append(2)
    P_try['P' + str(3)].append(3)
    P_try['P' + str(0)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(1)].append(1)
    P_try['P' + str(2)].append(2)
    P_try['P' + str(3)].append(3)
    P_try['P' + str(1)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(1)].append(1)
    P_try['P' + str(2)].append(2)
    P_try['P' + str(3)].append(3)
    P_try['P' + str(2)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(1)].append(1)
    P_try['P' + str(2)].append(2)
    P_try['P' + str(3)].append(3)
    P_try['P' + str(3)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(1)].append(1)
    P_try['P' + str(3)].append(2)
    P_try['P' + str(0)].append(3)
    P_try['P' + str(0)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(1)].append(1)
    P_try['P' + str(3)].append(2)
    P_try['P' + str(0)].append(3)
    P_try['P' + str(1)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(1)].append(1)
    P_try['P' + str(3)].append(2)
    P_try['P' + str(0)].append(3)
    P_try['P' + str(2)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(1)].append(1)
    P_try['P' + str(3)].append(2)
    P_try['P' + str(0)].append(3)
    P_try['P' + str(3)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(1)].append(1)
    P_try['P' + str(3)].append(2)
    P_try['P' + str(1)].append(3)
    P_try['P' + str(0)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(1)].append(1)
    P_try['P' + str(3)].append(2)
    P_try['P' + str(1)].append(3)
    P_try['P' + str(1)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(1)].append(1)
    P_try['P' + str(3)].append(2)
    P_try['P' + str(1)].append(3)
    P_try['P' + str(2)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(1)].append(1)
    P_try['P' + str(3)].append(2)
    P_try['P' + str(1)].append(3)
    P_try['P' + str(3)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(1)].append(1)
    P_try['P' + str(3)].append(2)
    P_try['P' + str(2)].append(3)
    P_try['P' + str(0)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(1)].append(1)
    P_try['P' + str(3)].append(2)
    P_try['P' + str(2)].append(3)
    P_try['P' + str(1)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(1)].append(1)
    P_try['P' + str(3)].append(2)
    P_try['P' + str(2)].append(3)
    P_try['P' + str(2)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(1)].append(1)
    P_try['P' + str(3)].append(2)
    P_try['P' + str(2)].append(3)
    P_try['P' + str(3)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(1)].append(1)
    P_try['P' + str(3)].append(2)
    P_try['P' + str(3)].append(3)
    P_try['P' + str(0)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(1)].append(1)
    P_try['P' + str(3)].append(2)
    P_try['P' + str(3)].append(3)
    P_try['P' + str(1)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(1)].append(1)
    P_try['P' + str(3)].append(2)
    P_try['P' + str(3)].append(3)
    P_try['P' + str(2)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(1)].append(1)
    P_try['P' + str(3)].append(2)
    P_try['P' + str(3)].append(3)
    P_try['P' + str(3)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(2)].append(1)
    P_try['P' + str(0)].append(2)
    P_try['P' + str(0)].append(3)
    P_try['P' + str(0)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(2)].append(1)
    P_try['P' + str(0)].append(2)
    P_try['P' + str(0)].append(3)
    P_try['P' + str(1)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(2)].append(1)
    P_try['P' + str(0)].append(2)
    P_try['P' + str(0)].append(3)
    P_try['P' + str(2)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(2)].append(1)
    P_try['P' + str(0)].append(2)
    P_try['P' + str(0)].append(3)
    P_try['P' + str(3)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(2)].append(1)
    P_try['P' + str(0)].append(2)
    P_try['P' + str(1)].append(3)
    P_try['P' + str(0)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(2)].append(1)
    P_try['P' + str(0)].append(2)
    P_try['P' + str(1)].append(3)
    P_try['P' + str(1)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(2)].append(1)
    P_try['P' + str(0)].append(2)
    P_try['P' + str(1)].append(3)
    P_try['P' + str(2)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(2)].append(1)
    P_try['P' + str(0)].append(2)
    P_try['P' + str(1)].append(3)
    P_try['P' + str(3)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(2)].append(1)
    P_try['P' + str(0)].append(2)
    P_try['P' + str(2)].append(3)
    P_try['P' + str(0)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(2)].append(1)
    P_try['P' + str(0)].append(2)
    P_try['P' + str(2)].append(3)
    P_try['P' + str(1)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(2)].append(1)
    P_try['P' + str(0)].append(2)
    P_try['P' + str(2)].append(3)
    P_try['P' + str(2)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(2)].append(1)
    P_try['P' + str(0)].append(2)
    P_try['P' + str(2)].append(3)
    P_try['P' + str(3)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(2)].append(1)
    P_try['P' + str(0)].append(2)
    P_try['P' + str(3)].append(3)
    P_try['P' + str(0)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(2)].append(1)
    P_try['P' + str(0)].append(2)
    P_try['P' + str(3)].append(3)
    P_try['P' + str(1)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(2)].append(1)
    P_try['P' + str(0)].append(2)
    P_try['P' + str(3)].append(3)
    P_try['P' + str(2)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(2)].append(1)
    P_try['P' + str(0)].append(2)
    P_try['P' + str(3)].append(3)
    P_try['P' + str(3)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(2)].append(1)
    P_try['P' + str(1)].append(2)
    P_try['P' + str(0)].append(3)
    P_try['P' + str(0)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(2)].append(1)
    P_try['P' + str(1)].append(2)
    P_try['P' + str(0)].append(3)
    P_try['P' + str(1)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(2)].append(1)
    P_try['P' + str(1)].append(2)
    P_try['P' + str(0)].append(3)
    P_try['P' + str(2)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(2)].append(1)
    P_try['P' + str(1)].append(2)
    P_try['P' + str(0)].append(3)
    P_try['P' + str(3)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(2)].append(1)
    P_try['P' + str(1)].append(2)
    P_try['P' + str(1)].append(3)
    P_try['P' + str(0)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(2)].append(1)
    P_try['P' + str(1)].append(2)
    P_try['P' + str(1)].append(3)
    P_try['P' + str(1)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(2)].append(1)
    P_try['P' + str(1)].append(2)
    P_try['P' + str(1)].append(3)
    P_try['P' + str(2)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(2)].append(1)
    P_try['P' + str(1)].append(2)
    P_try['P' + str(1)].append(3)
    P_try['P' + str(3)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(2)].append(1)
    P_try['P' + str(1)].append(2)
    P_try['P' + str(2)].append(3)
    P_try['P' + str(0)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(2)].append(1)
    P_try['P' + str(1)].append(2)
    P_try['P' + str(2)].append(3)
    P_try['P' + str(1)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(2)].append(1)
    P_try['P' + str(1)].append(2)
    P_try['P' + str(2)].append(3)
    P_try['P' + str(2)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(2)].append(1)
    P_try['P' + str(1)].append(2)
    P_try['P' + str(2)].append(3)
    P_try['P' + str(3)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(2)].append(1)
    P_try['P' + str(1)].append(2)
    P_try['P' + str(3)].append(3)
    P_try['P' + str(0)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(2)].append(1)
    P_try['P' + str(1)].append(2)
    P_try['P' + str(3)].append(3)
    P_try['P' + str(1)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(2)].append(1)
    P_try['P' + str(1)].append(2)
    P_try['P' + str(3)].append(3)
    P_try['P' + str(2)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(2)].append(1)
    P_try['P' + str(1)].append(2)
    P_try['P' + str(3)].append(3)
    P_try['P' + str(3)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(2)].append(1)
    P_try['P' + str(2)].append(2)
    P_try['P' + str(0)].append(3)
    P_try['P' + str(0)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(2)].append(1)
    P_try['P' + str(2)].append(2)
    P_try['P' + str(0)].append(3)
    P_try['P' + str(1)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(2)].append(1)
    P_try['P' + str(2)].append(2)
    P_try['P' + str(0)].append(3)
    P_try['P' + str(2)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(2)].append(1)
    P_try['P' + str(2)].append(2)
    P_try['P' + str(0)].append(3)
    P_try['P' + str(3)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(2)].append(1)
    P_try['P' + str(2)].append(2)
    P_try['P' + str(1)].append(3)
    P_try['P' + str(0)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(2)].append(1)
    P_try['P' + str(2)].append(2)
    P_try['P' + str(1)].append(3)
    P_try['P' + str(1)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(2)].append(1)
    P_try['P' + str(2)].append(2)
    P_try['P' + str(1)].append(3)
    P_try['P' + str(2)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(2)].append(1)
    P_try['P' + str(2)].append(2)
    P_try['P' + str(1)].append(3)
    P_try['P' + str(3)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(2)].append(1)
    P_try['P' + str(2)].append(2)
    P_try['P' + str(2)].append(3)
    P_try['P' + str(0)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(2)].append(1)
    P_try['P' + str(2)].append(2)
    P_try['P' + str(2)].append(3)
    P_try['P' + str(1)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(2)].append(1)
    P_try['P' + str(2)].append(2)
    P_try['P' + str(2)].append(3)
    P_try['P' + str(2)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(2)].append(1)
    P_try['P' + str(2)].append(2)
    P_try['P' + str(2)].append(3)
    P_try['P' + str(3)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(2)].append(1)
    P_try['P' + str(2)].append(2)
    P_try['P' + str(3)].append(3)
    P_try['P' + str(0)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(2)].append(1)
    P_try['P' + str(2)].append(2)
    P_try['P' + str(3)].append(3)
    P_try['P' + str(1)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(2)].append(1)
    P_try['P' + str(2)].append(2)
    P_try['P' + str(3)].append(3)
    P_try['P' + str(2)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(2)].append(1)
    P_try['P' + str(2)].append(2)
    P_try['P' + str(3)].append(3)
    P_try['P' + str(3)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(2)].append(1)
    P_try['P' + str(3)].append(2)
    P_try['P' + str(0)].append(3)
    P_try['P' + str(0)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(2)].append(1)
    P_try['P' + str(3)].append(2)
    P_try['P' + str(0)].append(3)
    P_try['P' + str(1)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(2)].append(1)
    P_try['P' + str(3)].append(2)
    P_try['P' + str(0)].append(3)
    P_try['P' + str(2)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(2)].append(1)
    P_try['P' + str(3)].append(2)
    P_try['P' + str(0)].append(3)
    P_try['P' + str(3)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(2)].append(1)
    P_try['P' + str(3)].append(2)
    P_try['P' + str(1)].append(3)
    P_try['P' + str(0)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(2)].append(1)
    P_try['P' + str(3)].append(2)
    P_try['P' + str(1)].append(3)
    P_try['P' + str(1)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(2)].append(1)
    P_try['P' + str(3)].append(2)
    P_try['P' + str(1)].append(3)
    P_try['P' + str(2)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(2)].append(1)
    P_try['P' + str(3)].append(2)
    P_try['P' + str(1)].append(3)
    P_try['P' + str(3)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(2)].append(1)
    P_try['P' + str(3)].append(2)
    P_try['P' + str(2)].append(3)
    P_try['P' + str(0)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(2)].append(1)
    P_try['P' + str(3)].append(2)
    P_try['P' + str(2)].append(3)
    P_try['P' + str(1)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(2)].append(1)
    P_try['P' + str(3)].append(2)
    P_try['P' + str(2)].append(3)
    P_try['P' + str(2)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(2)].append(1)
    P_try['P' + str(3)].append(2)
    P_try['P' + str(2)].append(3)
    P_try['P' + str(3)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(2)].append(1)
    P_try['P' + str(3)].append(2)
    P_try['P' + str(3)].append(3)
    P_try['P' + str(0)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(2)].append(1)
    P_try['P' + str(3)].append(2)
    P_try['P' + str(3)].append(3)
    P_try['P' + str(1)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(2)].append(1)
    P_try['P' + str(3)].append(2)
    P_try['P' + str(3)].append(3)
    P_try['P' + str(2)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(2)].append(1)
    P_try['P' + str(3)].append(2)
    P_try['P' + str(3)].append(3)
    P_try['P' + str(3)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(3)].append(1)
    P_try['P' + str(0)].append(2)
    P_try['P' + str(0)].append(3)
    P_try['P' + str(0)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(3)].append(1)
    P_try['P' + str(0)].append(2)
    P_try['P' + str(0)].append(3)
    P_try['P' + str(1)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(3)].append(1)
    P_try['P' + str(0)].append(2)
    P_try['P' + str(0)].append(3)
    P_try['P' + str(2)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(3)].append(1)
    P_try['P' + str(0)].append(2)
    P_try['P' + str(0)].append(3)
    P_try['P' + str(3)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(3)].append(1)
    P_try['P' + str(0)].append(2)
    P_try['P' + str(1)].append(3)
    P_try['P' + str(0)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(3)].append(1)
    P_try['P' + str(0)].append(2)
    P_try['P' + str(1)].append(3)
    P_try['P' + str(1)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(3)].append(1)
    P_try['P' + str(0)].append(2)
    P_try['P' + str(1)].append(3)
    P_try['P' + str(2)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(3)].append(1)
    P_try['P' + str(0)].append(2)
    P_try['P' + str(1)].append(3)
    P_try['P' + str(3)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(3)].append(1)
    P_try['P' + str(0)].append(2)
    P_try['P' + str(2)].append(3)
    P_try['P' + str(0)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(3)].append(1)
    P_try['P' + str(0)].append(2)
    P_try['P' + str(2)].append(3)
    P_try['P' + str(1)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(3)].append(1)
    P_try['P' + str(0)].append(2)
    P_try['P' + str(2)].append(3)
    P_try['P' + str(2)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(3)].append(1)
    P_try['P' + str(0)].append(2)
    P_try['P' + str(2)].append(3)
    P_try['P' + str(3)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(3)].append(1)
    P_try['P' + str(0)].append(2)
    P_try['P' + str(3)].append(3)
    P_try['P' + str(0)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(3)].append(1)
    P_try['P' + str(0)].append(2)
    P_try['P' + str(3)].append(3)
    P_try['P' + str(1)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(3)].append(1)
    P_try['P' + str(0)].append(2)
    P_try['P' + str(3)].append(3)
    P_try['P' + str(2)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(3)].append(1)
    P_try['P' + str(0)].append(2)
    P_try['P' + str(3)].append(3)
    P_try['P' + str(3)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(3)].append(1)
    P_try['P' + str(1)].append(2)
    P_try['P' + str(0)].append(3)
    P_try['P' + str(0)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(3)].append(1)
    P_try['P' + str(1)].append(2)
    P_try['P' + str(0)].append(3)
    P_try['P' + str(1)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(3)].append(1)
    P_try['P' + str(1)].append(2)
    P_try['P' + str(0)].append(3)
    P_try['P' + str(2)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(3)].append(1)
    P_try['P' + str(1)].append(2)
    P_try['P' + str(0)].append(3)
    P_try['P' + str(3)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(3)].append(1)
    P_try['P' + str(1)].append(2)
    P_try['P' + str(1)].append(3)
    P_try['P' + str(0)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(3)].append(1)
    P_try['P' + str(1)].append(2)
    P_try['P' + str(1)].append(3)
    P_try['P' + str(1)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(3)].append(1)
    P_try['P' + str(1)].append(2)
    P_try['P' + str(1)].append(3)
    P_try['P' + str(2)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(3)].append(1)
    P_try['P' + str(1)].append(2)
    P_try['P' + str(1)].append(3)
    P_try['P' + str(3)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(3)].append(1)
    P_try['P' + str(1)].append(2)
    P_try['P' + str(2)].append(3)
    P_try['P' + str(0)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(3)].append(1)
    P_try['P' + str(1)].append(2)
    P_try['P' + str(2)].append(3)
    P_try['P' + str(1)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(3)].append(1)
    P_try['P' + str(1)].append(2)
    P_try['P' + str(2)].append(3)
    P_try['P' + str(2)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(3)].append(1)
    P_try['P' + str(1)].append(2)
    P_try['P' + str(2)].append(3)
    P_try['P' + str(3)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(3)].append(1)
    P_try['P' + str(1)].append(2)
    P_try['P' + str(3)].append(3)
    P_try['P' + str(0)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(3)].append(1)
    P_try['P' + str(1)].append(2)
    P_try['P' + str(3)].append(3)
    P_try['P' + str(1)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(3)].append(1)
    P_try['P' + str(1)].append(2)
    P_try['P' + str(3)].append(3)
    P_try['P' + str(2)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(3)].append(1)
    P_try['P' + str(1)].append(2)
    P_try['P' + str(3)].append(3)
    P_try['P' + str(3)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(3)].append(1)
    P_try['P' + str(2)].append(2)
    P_try['P' + str(0)].append(3)
    P_try['P' + str(0)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(3)].append(1)
    P_try['P' + str(2)].append(2)
    P_try['P' + str(0)].append(3)
    P_try['P' + str(1)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(3)].append(1)
    P_try['P' + str(2)].append(2)
    P_try['P' + str(0)].append(3)
    P_try['P' + str(2)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(3)].append(1)
    P_try['P' + str(2)].append(2)
    P_try['P' + str(0)].append(3)
    P_try['P' + str(3)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(3)].append(1)
    P_try['P' + str(2)].append(2)
    P_try['P' + str(1)].append(3)
    P_try['P' + str(0)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(3)].append(1)
    P_try['P' + str(2)].append(2)
    P_try['P' + str(1)].append(3)
    P_try['P' + str(1)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(3)].append(1)
    P_try['P' + str(2)].append(2)
    P_try['P' + str(1)].append(3)
    P_try['P' + str(2)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(3)].append(1)
    P_try['P' + str(2)].append(2)
    P_try['P' + str(1)].append(3)
    P_try['P' + str(3)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(3)].append(1)
    P_try['P' + str(2)].append(2)
    P_try['P' + str(2)].append(3)
    P_try['P' + str(0)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(3)].append(1)
    P_try['P' + str(2)].append(2)
    P_try['P' + str(2)].append(3)
    P_try['P' + str(1)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(3)].append(1)
    P_try['P' + str(2)].append(2)
    P_try['P' + str(2)].append(3)
    P_try['P' + str(2)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(3)].append(1)
    P_try['P' + str(2)].append(2)
    P_try['P' + str(2)].append(3)
    P_try['P' + str(3)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(3)].append(1)
    P_try['P' + str(2)].append(2)
    P_try['P' + str(3)].append(3)
    P_try['P' + str(0)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(3)].append(1)
    P_try['P' + str(2)].append(2)
    P_try['P' + str(3)].append(3)
    P_try['P' + str(1)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(3)].append(1)
    P_try['P' + str(2)].append(2)
    P_try['P' + str(3)].append(3)
    P_try['P' + str(2)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(3)].append(1)
    P_try['P' + str(2)].append(2)
    P_try['P' + str(3)].append(3)
    P_try['P' + str(3)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(3)].append(1)
    P_try['P' + str(3)].append(2)
    P_try['P' + str(0)].append(3)
    P_try['P' + str(0)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(3)].append(1)
    P_try['P' + str(3)].append(2)
    P_try['P' + str(0)].append(3)
    P_try['P' + str(1)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(3)].append(1)
    P_try['P' + str(3)].append(2)
    P_try['P' + str(0)].append(3)
    P_try['P' + str(2)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(3)].append(1)
    P_try['P' + str(3)].append(2)
    P_try['P' + str(0)].append(3)
    P_try['P' + str(3)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(3)].append(1)
    P_try['P' + str(3)].append(2)
    P_try['P' + str(1)].append(3)
    P_try['P' + str(0)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(3)].append(1)
    P_try['P' + str(3)].append(2)
    P_try['P' + str(1)].append(3)
    P_try['P' + str(1)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(3)].append(1)
    P_try['P' + str(3)].append(2)
    P_try['P' + str(1)].append(3)
    P_try['P' + str(2)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(3)].append(1)
    P_try['P' + str(3)].append(2)
    P_try['P' + str(1)].append(3)
    P_try['P' + str(3)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(3)].append(1)
    P_try['P' + str(3)].append(2)
    P_try['P' + str(2)].append(3)
    P_try['P' + str(0)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(3)].append(1)
    P_try['P' + str(3)].append(2)
    P_try['P' + str(2)].append(3)
    P_try['P' + str(1)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(3)].append(1)
    P_try['P' + str(3)].append(2)
    P_try['P' + str(2)].append(3)
    P_try['P' + str(2)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(3)].append(1)
    P_try['P' + str(3)].append(2)
    P_try['P' + str(2)].append(3)
    P_try['P' + str(3)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(3)].append(1)
    P_try['P' + str(3)].append(2)
    P_try['P' + str(3)].append(3)
    P_try['P' + str(0)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(3)].append(1)
    P_try['P' + str(3)].append(2)
    P_try['P' + str(3)].append(3)
    P_try['P' + str(1)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(3)].append(1)
    P_try['P' + str(3)].append(2)
    P_try['P' + str(3)].append(3)
    P_try['P' + str(2)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try
    P_try = {'P0': [0], 'P1': [], 'P2': [], 'P3': []}
    P_try['P' + str(3)].append(1)
    P_try['P' + str(3)].append(2)
    P_try['P' + str(3)].append(3)
    P_try['P' + str(3)].append(4)
    C_try = cost(G, P_try)
    if C_try > C_opt:
        P = P_try.copy()
        C_opt = C_try

    return C_opt, P


def cost(G, P):
    """Cost of a given partition P of Graph G
    Args:
        G (graph): graph the partition is defined on
        P (dict): dictionary with changing points as keys and array of jobs (nodes) for the charging point as values
    Returns:
        C (int): cost function of the given Partition
    """
    partitions = list(P.keys())
    C = 0
    for Pr in partitions:
        for Ps in partitions[partitions.index(Pr) + 1:]:
            C = C + cut_weight(G, P[Pr], P[Ps])
            # print('Pr=',Pr , ' Ps=', Ps, '   cut_weight=', cut_weight(G, P[Pr], P[Ps]), '\n')
    return C


def cut_weight(G, P1, P2):
    """weight of all edges connecting the subnodesets P1 and P2 in Graph G
    Args:
        G (graph): graph the partition is defined on
        P1 (dict): nodes of the first subset
        P2 (dict): nodes of the second subset
    Returns:
        w (int): sum of weights with one node in P1 and the other in P2
    """
    weights = get_weight_matrix(G)
    w = np.sum([weights[i, j] for i in P1 for j in P2])
    return w


def rec_cost_optimization(G, k, N, N_rec, P_opt, C_opt):
    """Recursive part of the classical brut-force solution for arbitrary k and N
    Args:
        G (graph): Graph of the max-k-cut problem
        k (int): Number of cuts (available loading stations)
        N (int): Number of nodes of Graph G
        N_rec(int): recursive variable, initially N_rec=N-1
        C_opt (int): cost function of optimal max-k-cut at the current opimization step solution
        P_opt (dict): dictionary with the number of the load station as keys and array of jobs at a given
                loading station (still unordered!) as values at the current opimization step solution
    Globals:
        time (int): Used for progress output
    Returns:
        C_opt (int): cost function of optimal max-k-cut solution
        P_opt (dict): dictionary with the number of the load station as keys
                and array of jobs at a given loading station (still unordered!) as values
    Method:
        recursive version of the for-loops in brut_force_k4_N5 for arbitrary k and N
    """
    if N_rec > 0:
        for globals()['PN%i' % N_rec] in range(k):
            if N_rec == 7:
                global time
                print('{}\r'.format('progress ' + str(np.round(1 / k ** (N - 7) * time * 100, 4)) + '%' + '   '),
                      end="")
                # '  P_opt='+ P_opt+ '   C_opt='+ C_opt)
                time += 1
            C_opt, P_opt = rec_cost_optimization(G, k, N, N_rec - 1, P_opt, C_opt)
    else:
        # PN0=0  # node o is always in Partition 0
        P_try = {'P0': [0]}
        for i in range(1, k):
            P_try['P' + str(i)] = []
        for j in range(1, N):
            P_try['P' + str(eval('PN' + str(j)))].append(j)
        C_try = cost(G, P_try)
        if C_try > C_opt:
            P_opt = P_try.copy()
            C_opt = C_try
    return C_opt, P_opt


def benchmarking(function, *args):
    start = dt.datetime.now()
    result = function(*args)
    stop = dt.datetime.now()
    passed_time = stop - start
    return result, passed_time


def str_array_into_dict(P_str_array):
    P_dict = {}
    participation_data = ast.literal_eval(P_str_array)
    P_len = len(participation_data)
    for i in range(P_len):  # Key muss array sein, kein string
        P_dict["P" + str(i)] = participation_data[i]
    return P_dict



def plot_distribution_diagramm(G, distribution_qaoa):
    count_list_qaoa = [[], []]
    for key, value in distribution_qaoa.items():
        P = str_array_into_dict(key)
        C = cost(G, P)
        count_list_qaoa[0].append(C)
        count_list_qaoa[1].append(value)

    plt.figure()
    plt.bar(count_list_qaoa[0], count_list_qaoa[1], width=50)



def qaoa_highest_counts(distribution_qaoa, number_highest_counts):
    count_list = list(distribution_qaoa.values())
    top_indices = np.argsort(count_list)[-number_highest_counts:]
    top_counts = [count_list[i] for i in top_indices]

    highest_distr_qaoa = dict()
    for key, value in distribution_qaoa.items():
        if value in top_counts:
            highest_distr_qaoa[key] = value

    return highest_distr_qaoa


# Quantum functions

def make_mixing_block(n, l, layer):
    """
    :param n: The number of nodes
    :param l: The number of qubits representing a node
    :param layer: The subscript to be given to the variable beta, the variable would be named to 'beta_layer'
    :return: The QuantumCircuit object corresponding to the mixing block
    """
    nq = n * l
    globals()['beta%s' % layer] = Parameter('\u03B2' + str(layer))
    mix_circ = QuantumCircuit(nq)
    for q in range(0, nq):
        mix_circ.rx(2 * globals()['beta%s' % layer], q)
    return mix_circ


def make_cost_block(n, l, w, layer):
    """
    :param n: The number of nodes
    :param l: The number of qubits representing a node
    :param w: The weight matrix for edges
    :param layer: The subscript to be given to the variable gamma, the variable would be named to 'gamma_layer'
    :return: The QuantumCircuit object corresponding to the cost block
    """
    nq = n * l
    globals()['gamma%s' % layer] = Parameter('\u03B3' + str(layer))
    cost_circ = QuantumCircuit(nq)
    for n1 in range(n):  # first node
        for n2 in range(n1 + 1, n):  # second node, which is always larger than first node by index to avoid repetition
            for s in powerset([i for i in range(l)]):  # generate a power set so that we can perform cx-rz-cx, cx-cx-rz-
                # cx-cx, etc. sub-blocks one by one
                if len(s) == 0:  # empty set, we could do a global phase rotation but unnecessary
                    pass
                else:
                    seq = [l * n1 + i for i in s] + [l * n2 + i for i in s]  # list out all the qubits involved for a
                    # sub-block
                    seq.sort()  # we want to apply cx gates in order
                    for i in range(len(seq) - 1):
                        cost_circ.cx(seq[i], seq[i + 1])  # apply cx gates in ascending order of index
                    cost_circ.rz(globals()['gamma%s' % layer] * w[n1][n2] / 2, seq[-1])  # apply the rz gate
                    for i in range(len(seq) - 1):
                        cost_circ.cx(seq[-(i + 2)], seq[-(i + 1)])  # apply cx gates in descending order of index
                #cost_circ.barrier()
    return cost_circ


def make_initial_block(n, l):
    """
    :param n: The number of nodes
    :param l: The number of qubits representing a node
    :return: The QuantumCircuit object corresponding to the initial state
    """
    nq = n * l
    init_circ = QuantumCircuit(nq)
    for i in range(0, nq):
        init_circ.h(i)
    return init_circ


def make_full_circuit(n, l, w, p):
    """
    :param n: The number of nodes
    :param l: The number of qubits representing a node
    :param w: The weight matrix for edges
    :param p: The number of layers
    :return: The QuantumCircuit object corresponding to the full circuit
    """
    nq = n * l
    circ = QuantumCircuit(nq)
    circ.append(make_initial_block(n, l), [i for i in range(nq)])
    for layer in range(p):
        circ.append(make_cost_block(n, l, w, layer), [i for i in range(nq)])
        circ.append(make_mixing_block(n, l, layer), [i for i in range(nq)])
    circ.measure_all()
    return circ


def run_circuit(circ, param_list, nshots=512, simulator='qasm_simulator'):
    """
    :param circ:  The QuantumCircuit object corresponding to the full circuit
    :param param_list:  A list containing the parameters for the circuit
    :param nshots:  The number of shots
    :param simulator:  The simulator
    :return:  counts = dict{measurement result in string:count},
              transpiled_circ = The QuantumCircuit object corresponding to the transpiled circuit
    """
    circ_w_param = circ.bind_parameters(param_list)
    backend = Aer.get_backend(simulator)
    transpiled_circ = transpile(circ_w_param, backend)
    counts = backend.run(transpiled_circ, shots=nshots).result().get_counts()
    return counts, transpiled_circ



def func_to_optimize_wrapper(circ, l, w, nshots=512, simulator='qasm_simulator'):
    """
    :param circ:  The QuantumCircuit object corresponding to the full circuit, without feeding parameters or transpiling
    :param l: The number of qubits representing a node
    :param w: The weight matrix for edges
    :param nshots:  The number of shots
    :param simulator:  The simulator
    :return:  The function to optimize, corresponding to the circuit, to be fed to scipy.optimization.minimize
    """
    backend = Aer.get_backend(simulator)
    backend.shots = nshots

    def func_to_optimize(param_list):
         circ_w_param = circ.bind_parameters(param_list)
         transpiled_circ = transpile(circ_w_param, backend)
         counts = backend.run(transpiled_circ, shots=nshots).result().get_counts()
         return -1 * compute_cost(counts, l, w, nshots)

    return func_to_optimize


def full_optimization_loop(n, l, w, p, bounds=[(-np.pi, np.pi), (0, 4*np.pi)], nshots=512, simulator='qasm_simulator',
                           local_optimization_method='BFGS', optimal_cost=None):
    """
    :param n: The number of nodes
    :param l: The number of qubits representing a node
    :param w: The weight matrix for edges
    :param p: The number of layers
    :param bounds: The bounds for parameters, in the form of [(param_1_low, param_1_high),
                                                              (param_2_low, param_2_high), ...]
    :param nshots: The number of shots
    :param simulator: The simulator
    :param local_optimization_method: The local optimizer to use. By default, BFGS.
                                      Other recommended choices include Powell, Nelder-Mead, COBYLA, ...
                                      When set to None, no local optimization will be performed
    :param optimal_cost: The optimal cost found by brute force. By default, None.
    :return: three lists showing optimization history results, containing respectively
             1. parameters in list format,
             2. cost in float
             3. ciruits run as Qiskit QuantumCircuit objects
    """
    ####################################################################################################################
    # Initialize
    backend = Aer.get_backend(simulator)
    backend.shots = nshots
    param_history = []
    cost_history = []
    circ_history = []
    ####################################################################################################################
    # Run the educated global guess (EGG) optimization for the first time
    circ = make_full_circuit(n, l, w, 1)
    circ_history.append(circ)
    func_to_optimize = func_to_optimize_wrapper(circ, l, w, nshots=nshots, simulator=simulator)
    result = differential_evolution(func_to_optimize, bounds)
    param, cost = result.x, -1 * result.fun
    print('1st params', param)
    print('1st cost', cost)
    param_history.append(param)
    cost_history.append(cost)
    ####################################################################################################################
    # If depth = 1, no need to continue
    if p < 2:
        return param_history, cost_history, circ_history
    ####################################################################################################################
    # Else continue
    for i in range(2, p+1):
        if i == 2:
            abbrev = 'nd'
        else:
            abbrev = 'th'
    ####################################################################################################################
    # Run the educated global guess (EGG) optimization for ith iteration
        circ = make_full_circuit(n, l, w, i)
        param_names = circ.parameters
        param_bind_dict = {}
        for j in range(i-1):
            param_prev = param_history[-1]
            param_bind_dict[param_names[j]] = param_prev[j]
            param_bind_dict[param_names[j + i]] = param_prev[j + i - 1]
        circ_w_param = circ.bind_parameters(param_bind_dict)
        circ_history.append(circ_w_param)
        func_to_optimize = func_to_optimize_wrapper(circ_w_param, l, w, nshots=nshots, simulator=simulator)
        result = differential_evolution(func_to_optimize, bounds)
        param, cost = result.x, -1 * result.fun
        complete_param = np.concatenate((param_prev[:i-1], np.array([param[0]]),
                                         param_prev[i-1:], np.array([param[1]])))
        print(str(i) + abbrev + ' iteration (EGG), params', complete_param)
        print(str(i) + abbrev + ' iteration (EGG),  cost', cost)
        param_history.append(complete_param)
        cost_history.append(cost)
    ####################################################################################################################
    # Run the local optimization of choice for ith iteration if needed
        if local_optimization_method is not None:
            func_to_optimize = func_to_optimize_wrapper(circ, l, w, nshots=nshots, simulator=simulator)
            result = minimize(func_to_optimize, complete_param, method=local_optimization_method)
            param, cost = result.x, -1 * result.fun
            print(str(i) + abbrev + ' iteration (' + local_optimization_method + '), params', param)
            print(str(i) + abbrev + ' iteration (' + local_optimization_method + '), cost', cost)
            param_history.append(param)
            cost_history.append(cost)
            circ_history.append(circ)
    ####################################################################################################################
    # If optimal cost found by brute force is provided, compute the approximation ratio evolution
    if optimal_cost is not None:
        print('Approximation Ratio Evolution ', cost_history / optimal_cost)
    ####################################################################################################################
    # lists of parameters, costs and quantum circuits are returned
    return param_history, cost_history, circ_history





def qaoa_run(G, l, p, local_optimization_method='Powell', nshots=512):
    n = G.number_of_nodes()
    k = l ** 2

    weights = get_weight_matrix(G)
    R = compute_normalization_scale(weights)
    rescaled_weights = weights / R

    param_history, cost_history, circ_history = full_optimization_loop(n, l, rescaled_weights, p,
                                                                       local_optimization_method=local_optimization_method)

    circ = make_full_circuit(n, l, rescaled_weights, p)
    counts, transpiled_circ = run_circuit(circ, param_history[-1], nshots=nshots)

    return show_distribution(counts, l), param_history



def qaoa_solver(G, k, p):
    l = np.log2(k)
    if l != int(l):
        print('Error: Qaoa only works if k=l**2 with integer-valued l')
        return 0
    else:
        l = int(l)

    distribution_qaoa, param_history = qaoa_run(G, l, p, local_optimization_method='Nelder-Mead')

    plot_distribution_diagramm(G, distribution_qaoa)  # Plot to check

    highest_counts = qaoa_highest_counts(distribution_qaoa, 10)

    C_opt_qaoa = 0
    for key, value in highest_counts.items():
        P = str_array_into_dict(key)
        C = cost(G, P)
        if C > C_opt_qaoa:
            C_opt_qaoa = C
            P_opt_qaoa = P.copy()

    return C_opt_qaoa, P_opt_qaoa, distribution_qaoa, param_history
