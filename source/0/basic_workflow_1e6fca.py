# https://github.com/obstjn/ws-qaoa-transfer/blob/841b427f1c71a4005db74d482bce1729fcbd8f7a/scripts/basic_workflow.py
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.algorithms import QAOA
from qiskit import Aer, execute, BasicAer

import networkx as nx
from networkx import Graph

import matplotlib.pyplot as plt
import os
import itertools

# custom functions
from circuit_generation import qaoa_circuit
from calculations import get_energy, get_energy_grid
from plotting import plot_energy, save_contents, node_color_mapping, edge_color_mapping


# Graph
G = Graph()
G.add_edges_from([(i,i+1) for i in range(5)] + [(5,0), (0,2), (1,4), (3,5)])
_edge = (1,3)  # edge to calculate energy from
# draw graph
nx.draw_kamada_kawai(G, with_labels=True)

# circuit creation
qaoa_qc = qaoa_circuit(G, apx_sol=apx_sol, eps=0.1)
# draw circuit
qaoa_qc.draw('mpl')  

# energy calculation
energy_grid = get_energy_grid(G, qaoa_qc, edge=_edge, samples=30)
# load grid from file
#energy_grid = np.load('./paper-graphs/4-reg/3_energy.npy')
# print specific energy
#print(get_energy(G, qaoa_qc, gamma=0.2*np.pi, beta=0.3*np.pi))
# save energy to file
np.save(f'./ws-energies/{name}_energy.npy', energy_grid)

# plotting
plot_energy(energy_grid, title=apx_sol)

#save_contents(G, qaoa_qc, energy_grid, name='3', folder='paper-graphs/4-reg')


