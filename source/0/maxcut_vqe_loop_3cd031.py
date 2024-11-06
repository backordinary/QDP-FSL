# https://github.com/007Axel/Maxcut-Hopfield-QAOA/blob/da1006873c8b1d226de328e2b8002679e3a5f2c9/maxcut_vqe_loop.py
# VQE loop plus brute force check

import numpy as np
import networkx as nx
import qiskit_optimization as qco
import numba
import random
import itertools


from qiskit import Aer

from qiskit.tools.visualization import plot_histogram
from qiskit.circuit.library import TwoLocal
from qiskit.optimization.applications.ising import max_cut, tsp #read docs on this
from qiskit.aqua.algorithms import VQE, NumPyMinimumEigensolver
from qiskit.aqua.components.optimizers import SPSA
from qiskit.aqua import aqua_globals
from qiskit.aqua import QuantumInstance
from qiskit.optimization.applications.ising.common import sample_most_likely
from qiskit.optimization.algorithms import MinimumEigenOptimizer
from qiskit.optimization.problems import QuadraticProgram

# setup aqua logging
import logging
from qiskit.aqua import set_qiskit_aqua_logging
# set_qiskit_aqua_logging(logging.DEBUG)  # choose INFO, DEBUG to see the log

# Porting my graph generator (updated)
def G(n, seed, probability, negative_p, sparse=True, w=None, r=True):
    """
    Generates a seeded random graph with n nodes. 
    probability between 0 and 1 to increase odds.
    negative_p is the probability of a negative weight.
    sparse means if we want a sparse or dense graph.
    w is the weight of the edges if wanted.
    r means if we want random weights or not.
    """
    random.seed(seed)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i, j in itertools.combinations(G.nodes, 2):
        if sparse == True:
            if random.random() > (1 - probability):
                if r == True:
                    G.add_edge(i, j, weight=random.random())
                    # realized that does not cover negative weights, added:
                    #randomly make the weight negative or positive
                else:
                    G.add_edge(i, j, weight=w)
        if sparse == False:
            #Means we want a dense network, so
            if r == True:
                G.add_edge(i, j, weight=random.random())
            else:
                G.add_edge(i, j, weight=w)
    
    adjacency_matrix = nx.adjacency_matrix(G)
    for i, j in itertools.combinations(G.nodes, 2):
        if random.random() > (1 - negative_p):
            adjacency_matrix[i, j] = -1 * adjacency_matrix[i, j]

    return G, adjacency_matrix


n = 10
GG , h = G(n, random.random(), 0.5, 0.5, sparse=True, r=True)

adjacency = nx.adjacency_matrix(GG).todense()
b = np.zeros(GG.number_of_nodes())
Q = adjacency


#Solving via brute force

w = Q
best_cost_brute = 0
print_everything = False
for b in range(2**n):
    x = [int(t) for t in reversed(list(bin(b)[2:].zfill(n)))]
    cost = 0
    for i in range(n):
        for j in range(n):
            cost = cost + w[i,j]*x[i]*(1-x[j])
    if best_cost_brute < cost:
        best_cost_brute = cost
        xbest_brute = x
    if print_everything: 
        print('case = ' + str(x)+ ' cost = ' + str(cost))

colors = ['r' if xbest_brute[i] == 0 else 'c' for i in range(n)]
print('\nBest solution = ' + str(xbest_brute) + ' cost = ' + str(best_cost_brute))


# Save the best brute solution to a CSV
with open('brutevcost.csv','a') as fd:
    fd.write(str(best_cost_brute)+ '\n')

#Qiskit has an ising hamiltonian generator!
qubitOp, offset = max_cut.get_operator(w)

#Ising Hamiltonian to "Quadratic Program" (need to read their api on this)
qp = QuadraticProgram()
qp.from_ising(qubitOp, offset)
qp.to_docplex().prettyprint()

# solving Quadratic Program using exact classical eigensolver
exact = MinimumEigenOptimizer(NumPyMinimumEigensolver())
result = exact.solve(qp)


#Making the Hamiltonian in its full form and getting the lowest eigenvalue and eigenvector
ee = NumPyMinimumEigensolver(qubitOp)
result = ee.run()

x = sample_most_likely(result.eigenstate)
print('max-cut objective:', result.eigenvalue.real + offset)
print('solution:', max_cut.get_graph_solution(x))
print('solution objective:', max_cut.max_cut_value(x, w))


# Now, solving with Variational Quantum Eigensolver
aqua_globals.random_seed = np.random.default_rng(123)
seed = 10598
backend = Aer.get_backend('statevector_simulator')
quantum_instance = QuantumInstance(backend, seed_simulator=seed, seed_transpiler=seed)

# construct VQE
spsa = SPSA(maxiter=300)
ry = TwoLocal(qubitOp.num_qubits, 'ry', 'cz', reps=5, entanglement='linear')
vqe = VQE(qubitOp, ry, spsa, quantum_instance=quantum_instance)

# run VQE
result = vqe.run(quantum_instance)

# print results
x = sample_most_likely(result.eigenstate)
print('time:', result.optimizer_time)
print('max-cut objective:', result.eigenvalue.real + offset)
print('solution:', max_cut.get_graph_solution(x))
print('solution objective:', max_cut.max_cut_value(x, w))

# plot results
# x is the solution


h = x

cost = 0
for i in range(n):
    for j in range(n):
        cost = cost + w[i,j]*h[i]*(1-h[j])


with open('vqecost.csv','a') as fd:
    fd.write(str(cost)+ '\n')


time_used = result.optimizer_time
with open('timevqe.csv','a') as fd:
    fd.write(str(time_used)+ '\n')