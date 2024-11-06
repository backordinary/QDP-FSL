# https://github.com/smitchaudhary/QAOA-MaxCut/blob/9a91a85cc16f49fa986ce1761e44b5edc132c676/show_matrices.py
import qaoa_graphs as graphs
from statevector_functions import *

# General
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pylab as pl

# Qiskit
from qiskit.providers.aer import StatevectorSimulator
from qiskit import QuantumCircuit, execute

# Optimizers
from scipy.optimize import minimize, differential_evolution

# Choose your fighter
G = graphs.two_nodes_graph()
n = len(G.nodes())
E = G.edges()

# Choose the arena
backend = StatevectorSimulator()

# Choose number of rounds
p = 1
#colors = ['g' for node in G.nodes()]
#nx.draw_networkx(G, node_color=colors)

def give_cost(G, gamma, beta, p=1):

    n = len(G.nodes())
    E = G.edges()

    QAOA = QuantumCircuit(n, n)
    QAOA.h(range(n))
    QAOA.barrier()

    #plt.savefig(f'Initial State bad')
    np.set_printoptions(precision=3, suppress=True)
    for i in range(p):
        for edge in E:
            k = edge[0]
            l = edge[1]
            QAOA.cu1(-2*gamma[i], k, l) #Controlled-Z gate with a -2*gamma phase
            QAOA.u1(gamma[i], k) #Rotation of gamma around the z axis
            QAOA.u1(gamma[i], l)

        statevector = execute(QAOA, backend = StatevectorSimulator()).result().get_statevector(QAOA)
        print(f'The amplitudes after applying U_C are :\n {statevector}')
        probabilities = np.array(([abs(i)**2 for i in statevector]))
        #print(f'The probabilities after applying U_C are :\n {probabilities}')
        state_dict = {bin(i)[2:].zfill(n) : probabilities[i] for i in range(len(probabilities))}

        QAOA.barrier()
        QAOA.rx(2*beta[i], range(n)) #X rotation
        statevector = execute(QAOA, backend = StatevectorSimulator()).result().get_statevector(QAOA)
        print(f'The amplitudes after applying U_b are :\n {statevector}')
        probabilities = np.array(([abs(i)**2 for i in statevector]))
        print(f'The probabilities after applying U_B are :\n {probabilities}')
        state_dict = {bin(i)[2:].zfill(n) : probabilities[i] for i in range(len(probabilities))}
        cost = get_expectval(state_dict, G)
        print(f'The cost is {cost}')
        print()
        QAOA.barrier()
    return cost
    #plt.show()

def show_matrix(G, gamma, beta, p=1, state=0):

    n = len(G.nodes())
    E = G.edges()

    QAOA = QuantumCircuit(n, n)
    if state == 1:
        QAOA.x(0)
    elif state == 2:
        QAOA.x(1)
    elif state == 3:
        QAOA.x(0)
        QAOA.x(1)
    #QAOA.h(range(n))
    #QAOA.barrier()

    #plt.savefig(f'Initial State bad')
    np.set_printoptions(precision=3, suppress=True)
    for i in range(p):
        for edge in E:
            k = edge[0]
            l = edge[1]
            QAOA.cu1(-2*gamma[i], k, l) #Controlled-Z gate with a -2*gamma phase
            QAOA.u1(gamma[i], k) #Rotation of gamma around the z axis
            QAOA.u1(gamma[i], l)

        b = []
        c = []
        c_p = []
        b_p = []
        statevector = execute(QAOA, backend = StatevectorSimulator()).result().get_statevector(QAOA)
        c = np.array(statevector[:])
        #print(f'The amplitudes after applying U_C are :\n {statevector}')
        probabilities = np.array(([abs(i)**2 for i in statevector]))
        c_p = np.array(probabilities[:])
        #print(f'The probabilities after applying U_C are :\n {probabilities}')
        state_dict = {bin(i)[2:].zfill(n) : probabilities[i] for i in range(len(probabilities))}

        QAOA.barrier()
        QAOA.rx(2*beta[i], range(n)) #X rotation
        statevector = execute(QAOA, backend = StatevectorSimulator()).result().get_statevector(QAOA)
        b = np.array(statevector[:])
        #print(f'The amplitudes after applying U_b are :\n {statevector}')
        probabilities = np.array(([abs(i)**2 for i in statevector]))
        b_p = np.array(probabilities[:])
        #print(f'The probabilities after applying U_B are :\n {probabilities}')
        state_dict = {bin(i)[2:].zfill(n) : probabilities[i] for i in range(len(probabilities))}

        QAOA.barrier()
    return c, b, c_p, b_p
    #plt.show()

#show_amplitudes(G, [-1.57080367], [0.39271383], p=p)

gammas = [[0], [2*1.57080367], [1.00456822], [-1.57080367], [-0.66634531]]
betas = [[0], [0.11244852], [1.12778854], [0.39271383], [0.5]]

overall_dict = {}

for i in range(len(gammas)):
    temp_dict = {}
    temp_dict['gamma'] = gammas[i]
    temp_dict['beta'] = betas[i]

    C_effect = np.zeros((4,4), dtype=np.complex_)
    B_effect = np.zeros((4,4), dtype=np.complex_)
    O_effect = np.zeros((4,4), dtype=np.complex_)

    for j in range(4):
        c, b, c_p, b_p = show_matrix(G, gammas[i], [0], p=p, state=j)
        for k in range(4):
            C_effect[j, k] = c[k]
    for j in range(4):
        c, b, c_p, b_p = show_matrix(G, [0], betas[i], p=p, state=j)
        for k in range(4):
            B_effect[j, k] = b[k]
    for j in range(4):
        c, b, c_p, b_p = show_matrix(G, gammas[i], betas[i], p = p, state = j)
        for k in range(4):
            O_effect[j, k] = b[k]

    cost = give_cost(G, gammas[i], betas[i])

    temp_dict['U_C'] = C_effect
    temp_dict['U_B'] = B_effect
    temp_dict['U_O'] = O_effect
    temp_dict['cost'] = cost
    temp_dict['c_p'] = c_p
    temp_dict['b_p'] = b_p

    overall_dict[i] = temp_dict

for i in range(len(overall_dict)):
    print(f'Iteration number {i+1}')
    print(f'Gamma is {overall_dict[i]["gamma"]}')
    print(f'Beta is {overall_dict[i]["beta"]}')
    #print(f'The effect of matrix U_C is\n {overall_dict[i]["U_C"]}')
    #print(f'The effect of matrix U_B is\n {overall_dict[i]["U_B"]}')
    #print(f'The effect of matrix U_B*U_C is\n {overall_dict[i]["U_O"]}')
    #print(f'The average cost with these beta and gamma is {overall_dict[i]["cost"]}')
    print('------------------------------------------------')
    print('')
