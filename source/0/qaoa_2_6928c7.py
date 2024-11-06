# https://github.com/allen900/max_cut_code/blob/936666743d8c07e2c0f6d0f320aad4adb720cb0d/qaoa/qaoa_2.py
# import math tools
import numpy as np
# from qiskit.visualization import plot_histogram
# import qiskit
from qiskit.tools.monitor import job_monitor
from qiskit.providers.ibmq import least_busy
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute
from qiskit import Aer, IBMQ
# from qiskit.providers.aer import QasmSimulator
from qiskit.visualization import plot_histogram

# We import the tools to handle general Graphs
import networkx as nx

# We import plotting tools
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import sys
sys.path.append(".")
from utils.cut import Cut
from utils.graph import Graph
# importing Qiskit
# print(qiskit.__version__)
# IBMQ.save_account(
#     '2d8e029396ed0cbfcb9bffa7323c9cdc827886cbc8546a3d192687f3c198374e5c3fe3e35aa9b0f916d42bfa21de25e8030342731b63c2fc884caed5481076bb')


def qaoa(graph):
    step_size = 0.1

    a_gamma = np.arange(0, np.pi, step_size)
    a_beta = np.arange(0, np.pi, step_size)
    a_gamma, a_beta = np.meshgrid(a_gamma, a_beta)

    F1 = 3-(np.sin(2*a_beta)**2*np.sin(2*a_gamma)**2-0.5 *
            np.sin(4*a_beta)*np.sin(4*a_gamma))*(1+np.cos(4*a_gamma)**2)

    result = np.where(F1 == np.amax(F1))
    a = list(zip(result[0], result[1]))[0]

    gamma = a[0]*step_size
    beta = a[1]*step_size

    # prepare the quantum and classical resisters
    QAOA = QuantumCircuit(len(graph.nodes()), len(graph.nodes()))

    # apply the layer of Hadamard gates to all qubits
    QAOA.h(range(len(graph.nodes())))
    QAOA.barrier()

    for edge in graph.edges():
        k = edge[0]
        l = edge[1]
        # print(k, l)
        QAOA.cp(-2*gamma, k, l)
        QAOA.p(gamma, k)
        QAOA.p(gamma, l)

    # then apply the single qubit X - rotations with angle beta to all qubits
    QAOA.barrier()
    QAOA.rx(2*beta, range(len(graph.nodes())))

    # Finally measure the result in the computational basis
    QAOA.barrier()
    QAOA.measure(range(len(graph.nodes())), range(len(graph.nodes())))

    backend = Aer.get_backend("qasm_simulator")
    shots = 10000

    simulate = execute(QAOA, backend=backend, shots=shots)
    QAOA_results = simulate.result()

    r_time = QAOA_results.time_taken/shots
    counts = QAOA_results.get_counts()

    avr_C = 0
    max_C = [0, 0]
    hist = {}

    for k in range(len(graph.edges())+1):
        hist[str(k)] = hist.get(str(k), 0)

    for sample in list(counts.keys()):

        # use sampled bit string x to compute C(x)
        x = [int(num) for num in list(sample)]
        tmp_eng = cost_function_C(x, graph)

        # compute the expectation value and energy distribution
        avr_C = avr_C + counts[sample]*tmp_eng
        hist[str(round(tmp_eng))] = hist.get(
            str(round(tmp_eng)), 0) + counts[sample]

        # save best bit string
        if(max_C[1] < tmp_eng):
            max_C[0] = sample
            max_C[1] = tmp_eng

    M1_sampled = avr_C/shots
    best = max_C[1]
    # print('\n --- SIMULATION RESULTS ---\n')
    # print('The sampled mean value is M1_sampled = %.02f while the true value is M1 = %.02f \n' %
    #     (M1_sampled, np.amax(F1)))
    # print('The approximate solution is x* = %s with C(x*) = %d \n' %
    #     (max_C[0], max_C[1]))
    # plt = plot_histogram(hist,figsize = (10,6),bar_labels = False,number_to_keep=20)
    # plt.savefig('test1.png')

    return best, M1_sampled, r_time


def cost_function_C(x, G):
    weighted = False
    E = G.edges()
    if(len(x) != len(G.nodes())):
        return np.nan

    C = 0
    w = 1
    for index in E:
        e1 = index[0]
        e2 = index[1]
        if weighted:
            w = G[e1][e2]['weight']   
        C = C + w*x[e1]*(1-x[e2]) + w*x[e2]*(1-x[e1])

    return C

if __name__ == '__main__':
    graph = Graph("data/musae_git_edges.csv", is_real=False)
    # for _ in range(10):
    b, avg, t = qaoa(graph.graph)
    print(b, avg, t)
    # print('------QAOA Performance------')
    # print(graph.s, 'nodes')
    # print("Expected size:", int(graph.s * graph.s * graph.b / 4))
    # print('Cut size:', quantum.evaluate_cut_size(graph.graph))
    # print('Running time:', duration)
