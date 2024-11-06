# https://github.com/koenmesman/qaoa/blob/3d08dae42fd8f32437bf34c6665a21bbfdb8ecaf/python/main.py
import numpy as np
import csv
from math import pi
from scipy.optimize import curve_fit
from qiskit import (
  QuantumCircuit,
  execute,
  Aer)
import time
import random
from matplotlib import cm
import matplotlib.pyplot as plt
plt.interactive(True)
backend = Aer.get_backend('qasm_simulator')
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize


# random graph generator
def generate_graph(n, r):
    nodes = list(range(0, n))
    e = int(round(n*r))
    edges = [[0]*2 for _ in range(e)]
    super_flag = 1
    timeout = time.time() + 60 * 3
    while super_flag:
        for x in range(e):
            flag = 1
            while flag:
                node_a = nodes
                edges[x][0] = random.choice(node_a[:(n-1)])
                node_b = nodes
                edges[x][1] = random.choice(node_b[edges[x][0] + 1:])
                if edges.count(edges[x]) > 1:
                    flag = 1
                else:
                    flag = 0
        super_flag = 0
        for a in nodes:
            if (sum(x.count(a) for x in edges)) == 0:
                super_flag = 1
            else:
                if time.time() > timeout:
                    print("Timeout error: Not all nodes could be connected in expected time. ",
                          "Please increase edge to vertice ratio or increase timeout value")
                    break
    return [n, edges]


def eval_cost(out_state):
    # evaluate Max-Cut
    v = graph[0]
    edges = graph[1]
    c = 0
    bin_len = "{0:0" + str(v) + "b}"  # string required for binary formatting
    bin_val = list(bin_len.format(list(out_state).index(max(out_state))))
    bin_val = [int(i) for i in bin_val]
    bin_val = [-1 if x == 0 else 1 for x in bin_val]

    for e in edges:
        c += 0.5 * (1 - int(bin_val[e[0]]) * int(bin_val[e[1]]))
    return c


def find_exp(graph, out_state):
    #evaluate the expectation value
    prob = list(out_state.values())
    states = list(out_state.keys())
    exp = 0
    for k in range(len(states)):
        exp += eval_cost(states[k])*prob[k]/qaoa_shots
    return exp


def qaoa_inv(params):
    states = qaoa_circ(params)
    #best = max(states, key=states.get)
    #out = eval_cost(best)
    out = find_exp(graph, states)
    if out == 0:
        inv = 2
    else:
        inv = 1/out
    return inv


# QAOA function
def qaoa_circ(params):
    beta, gamma = params
    v, edge_list = graph

    if beta < 0 or beta > (2*pi) or gamma < 0 or gamma > (2*pi):
        return 0
    else:
        qc = QuantumCircuit(v, v)
        for qubit in range(v):
            qc.h(qubit)
        for iteration in range(p):
            for e in edge_list:                     # TODO: fix for unordered edges e.g. (2,0)
                qc.cnot(e[0], e[1])
                qc.rz(-gamma, e[1])
                qc.cnot(e[0], e[1])
            for qb in vertice_list:
                qc.rx(2*beta, qb)
        qc.measure(range(v), range(v))
        result = execute(qc, backend, shots=qaoa_shots).result()
        out_state = result.get_counts()
        #print(qc)

    return out_state


def qaoa_landscape(g, n):

    betas = np.linspace(0, 2 * pi, n)
    gammas = np.linspace(0, 2 * pi, n)
    val = [[0]*n for _ in range(n)]
    for b in range(n):
        beta = betas[b]
        for c in range(n):
            gamma = gammas[c]
            val[b][c] = find_exp(g, qaoa_circ([beta, gamma], g))

            #print(val[b][c])
    X, Y = np.meshgrid(betas, gammas)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, np.array(val), cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
    plt.show()
    return


def line(x, a, b):
    return a * x + b


######################################################################################
ratio = 1
p = 1
qaoa_shots = 10000
data_shots = 5
max_size = 15

#############################################################
#qaoa_landscape(graph, 40)  # nr_nodes, edge list, resolution
d_time = []
data_points = []
data_x = []

problem_range = range(3, max_size)

for problem_size in problem_range:   # loop over problem size
    vertice_list = list(range(0, problem_size, 1))
    # generate a (pseudo) random graph
    graph = generate_graph(problem_size, ratio)
    edge_list = graph[1]

    for d in range(data_shots):    # loop over data points per problem size
        init_param = [pi, pi]
        start = time.time()
        result = minimize(qaoa_inv, init_param, method='nelder-mead',
                       options={'ftol': 1e-2, 'maxfev':400, 'disp': False}) #
        d_time.append(time.time() - start)
        print("now finished size ", problem_size, " number ", d)
        data_x.append(problem_size)

np.savetxt('nm_data.dat', [data_x, d_time])

#for problem_size in problem_range:   # loop over problem size
#    vertice_list = list(range(0, problem_size, 1))
#    # generate a (pseudo) random graph
#    graph = generate_graph(problem_size, ratio)
#    edge_list = graph[1]

#    for d in range(data_shots):    # loop over data points per problem size
#        init_param = [pi, pi]
#        start = time.time()
#        result = minimize(qaoa_inv, init_param, method='BFGS',
#                       options={'disp': False}) #
#        d_time.append(time.time() - start)
#        print("now finished size ", problem_size, " number ", d)
#        data_x.append(problem_size)

#np.savetxt('BFGS_data.dat', [data_x, d_time])
