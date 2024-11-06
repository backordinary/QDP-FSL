# https://github.com/Chinmaysul/Quantum-TSP/blob/1e07cd2a436ad999cbc39db0628d38a8397757ef/tsp_ising.py
from qiskit_optimization.applications import Maxcut, Tsp
# from qiskit import IBMQ
from qiskit import *
from time import *
import numpy as np
import random
from sys import maxsize
from itertools import permutations
import random
from time import *
import networkx as nx
from qiskit.algorithms import VQE, NumPyMinimumEigensolver
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit.utils import algorithm_globals, QuantumInstance
from qiskit.circuit.library import TwoLocal
from qiskit.algorithms.optimizers import SPSA
v=int(input("Number of vertices:")) 
e=v*(v-1)/2
graph = []
# matrix representation of graph
for i in range(v):
    for j in range(i+1,v):
        c=random.randint(1,20)
        graph.append((i,j,c))
g = nx.Graph()
g.add_weighted_edges_from(graph)
tsp=Tsp(g)
adj_matrix = nx.to_numpy_matrix(tsp.graph)
print("solution objective:", tsp.tsp_value(range(v), adj_matrix))