# https://github.com/MarsherSusanin/Hackathon_2021/blob/366cfca563843ac9f825b4876333b451b3b6ab2b/Quantum%20part/quantum_final_2.py
#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().run_line_magic('matplotlib', 'inline')
# Importing standard Qiskit libraries and configuring account
from qiskit import QuantumCircuit, execute, Aer, IBMQ
from qiskit.compiler import transpile, assemble
from qiskit.tools.jupyter import *
from qiskit.visualization import *
#visualization tools
import matplotlib.pyplot as plt
import matplotlib.axes as axes
#other tool
import numpy as np
import networkx as nx
from itertools import permutations
#quadratic program
from qiskit.optimization import QuadraticProgram
#TSP libraries
from qiskit.optimization.applications.ising import tsp
from qiskit.optimization.applications.ising.common import sample_most_likely
#quantum computing optimization
#from qiskit.optimization.converters import IsingToQuadraticProgram
from qiskit.aqua.algorithms import VQE, QAOA, NumPyMinimumEigensolver
from qiskit.optimization.algorithms import MinimumEigenOptimizer


# In[2]:


#function for solving the TSP with brute force, i.e. generate all permutations and calc distances
def brute_force_tsp(w):
    N = len(w)
    #generate tuples with all permutation of numbers 1,2...N-1
    #first index is zero but we want to start our travel in the first city (i.e. with index 0)
    a = list(permutations(range(1,N)))
    
    best_dist = 1e10 #distance at begining
    
    for i in a: #for all permutations
        distance = 0
        pre_j = 0 #starting in city 0
        for j in i: #for each element of a permutation
            distance = distance + w[pre_j,j] #going from one city to another
            pre_j = j #save previous city
        distance = distance + w[pre_j,0] #going back to city 0
        order = (0,) + i #route description (i is permutation, 0 at the begining - the first city)
        print('Order: ', order, ' Distance: ', distance) #show solutions
        if distance < best_dist:
            best_dist = distance
            best_order = order           
        
    print('Route length: ', best_dist)
    print('Route: ', best_order)    
    
    return best_dist, best_order

#showing resulting route in graph
def show_tsp_graph(route):
    n = len(route)
    #showing the route in graph
    G = nx.Graph() #graph
    G.add_nodes_from(range(0,n)) #add nodes
    #adding edges based on solution    
    for i in range(0,n-1):
        G.add_edge(route[i], route[i+1])
    G.add_edge(route[n-1], 0)
    nx.draw_networkx(G) #show graph

#decoding binary output of QAOA to actual solution
def decodeQAOAresults(res):
    n = int(len(res)**0.5)
    results = np.zeros(n)
    k = 0
    for i in range(0,n): #each n elements refers to one time point i
        for j in range(0,n): #in each time points there are all cities
            #when x = 1 then the city j is visited in ith time point
            if res[k] == 1: results[i] = j
            k = k + 1
    return results

def tspQuantumSolver(distances, backendName):
    citiesNumber = len(distances)
    coordinates = np.zeros([citiesNumber, 2])
    for i in range(0, citiesNumber): coordinates[i][0] = i + 1
    
    tspTask = tsp.TspData(name = 'TSP', dim = citiesNumber, w = distances, coord = coordinates)
    
    isingHamiltonian, offset = tsp.get_operator(tspTask)
    
    tspQubo = QuadraticProgram()
    tspQubo.from_ising(isingHamiltonian, offset)
    
    quantumProcessor = Aer.backends(name = backendName)[0]
    qaoa = MinimumEigenOptimizer(QAOA(quantum_instance = quantumProcessor))
    results = qaoa.solve(tspQubo)
    print('Route length: ', results.fval)
    route = decodeQAOAresults(results.x)
    print('Route: ', route)
    
    return results.fval, route



# In[3]:


# Generating a graph of 3 nodes
n = 5
num_qubits = n ** 2
G=nx.Graph()
G.add_nodes_from(np.arange(0,n,1))
elist=[(0,1,15.0),(1,2,3.0),(2,3,3.0),(3,4,12.0),(4,0,17.0)]
G.add_weighted_edges_from(elist)
       
#ins = Tsp.create_random_instance(n, seed=123)
#ins = tsp.random_tsp(n, seed=123)
#ins.graph.graph = G.graph

#qp = ins.to_quadratic_program()
#print(qp.export_as_lp_string())
#print('distance\n', ins.w)

#pos = {k: v for k, v in enumerate(ins.coord)}
colors = ['r' for node in G.nodes()]
pos = nx.spring_layout(G)

def draw_graph(G, colors, pos):
    default_axes = plt.axes(frameon=True)
    nx.draw_networkx(G, node_color=colors, node_size=600, alpha=.8, ax=default_axes, pos=pos)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels)

draw_graph(G, colors, pos)


# In[4]:


# Создаем матрицу смежности:
w = np.zeros([n,n])
for i in range(n):
    for j in range(n):
        temp = G.get_edge_data(i,j,default=0)
        if temp != 0:
            w[i,j] = temp['weight']
print(w)


# In[5]:


#distMatrix = np.array([[0,207,92,131],
#                       [207,0,300,350],
#                       [92,300,0,82],
#                       [131,350,82,0]
#                       ])

distMatrix = w

#brute force solution
lengthBrute, routeBrute = brute_force_tsp(distMatrix)
show_tsp_graph(routeBrute)


# In[ ]:


#quantum solution
lengthQuantum, routeQuantum = tspQuantumSolver(distMatrix, 'qasm_simulator')
show_tsp_graph(routeQuantum)

