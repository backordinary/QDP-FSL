# https://github.com/Chinmaysul/Quantum-TSP/blob/1e07cd2a436ad999cbc39db0628d38a8397757ef/tsp_q.py
from qiskit_optimization.applications import Maxcut, Tsp
# from qiskit import IBMQ
from qiskit import *
from time import *
import numpy as np
import random
# Python3 program to implement traveling salesman
# problem using naive approach.
from sys import maxsize
from itertools import permutations
import random
from time import *
import christofides
# IBMQ.save_account('ed25329f92efdae33706376b5c476e504817665ea73f1c000ba3868cf0e561f08286ab396578b6092359629a6f9d8b93539e8fbec05413691f3f0ae37c443566')
# IBMQ.load_account()
# implementation of traveling Salesman Problem
def travellingSalesmanProblem(graph, s,V):

	# store all vertex apart from source vertex
	vertex = []
	for i in range(V):
		if i != s:
			vertex.append(i)

	# store minimum weight Hamiltonian Cycle
	min_path = maxsize
	next_permutation=permutations(vertex)
	for i in next_permutation:

		# store current Path weight(cost)
		current_pathweight = 0

		# compute current path weight
		k = s
		for j in i:
			current_pathweight += graph[k][j]
			k = j
		current_pathweight += graph[k][s]

		# update minimum
		min_path = min(min_path, current_pathweight)
		
	return min_path




# Driver Code
v=int(input("Number of vertices:")) 
e=v*(v-1)/2
graph = [[maxsize for column in range(v)]
                for row in range(v)]
# matrix representation of graph
for i in range(v):
    for j in range(i+1,v):
        c=random.randint(1,20)
        graph[i][j]=c
        graph[j][i]=c
# a=travellingSalesmanProblem(graph,0,v)
b=int(Tsp.tsp_value(range(v),np.array(graph)))
TSP = christofides.compute(graph)
c=TSP['Travel_Cost']
# if(a==b):
#     print("success_1")
# else:
#     print("failure_1")
# if(a==c):
#     print("success_2")
# else:
#     print("failure_2")
print(b,c)
