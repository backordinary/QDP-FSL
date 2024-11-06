# https://github.com/Mephphisto/QAOA-Demo/blob/39510e425aebc943b43b364add978855475e9061/app.py
from QAOA_MAX_cut import MAX_Cut, QAOA_Test
import qiskit as qs
import networkx as nx
import matplotlib.pyplot as plt

#test funtionality with a random graph
QAOA_Test()

#Create graph PLEASE Feel free to replace
Graph = nx.erdos_renyi_graph(5, 0.75)
#Plot Graph
nx.draw(Graph)
plt.show()
#Insert Backend of choice
backend = qs.Aer.get_backend('qasm_simulator')

# Call to QAOA library
cut, solution = MAX_Cut(Graph, backend)
print(" best cut ", cut)
print(" Best Solution", solution)

#Make list of left nodes
left = []
for node in Graph:
    if solution[node] == '0':
        left.append(node)

#Plot with cut vertically
nx.draw(Graph,  pos=nx.bipartite_layout(Graph, left))
plt.show()