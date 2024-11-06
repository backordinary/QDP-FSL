# https://github.com/obstjn/ws-qaoa-transfer/blob/cc9d9be51485ebe2a195a4f157e5a4f33faa6978/scripts/ws_canceling.py
import numpy as np
import networkx as nx
from networkx import Graph
import matplotlib.pyplot as plt

from calculations import *
from circuit_generation import *
from plotting import *
from graph_management import *

from qiskit.visualization import plot_histogram
from qiskit import Aer, execute


G5reg0 = nx.read_adjlist('./graphs/5reg0.graph', nodetype=int)
G5reg0.remove_nodes_from([6,7,8,9])

draw_graph_with_ws(G5reg0, warmstarting=[0, 0, 0, 0, 1, 0])

qc = qaoa_circuit(G5reg0, apx_sol=[0, 0, 0, 0, 1, 0])

energy = get_energy_grid(G5reg0, qc, (4,5), samples=30)

#qc.draw('mpl')
#aG3reg0 = nx.read_adjlist('./graphs/3reg0.graph', nodetype=int)
#aqc = qaoa_circuit(aG3reg0, apx_sol=[1, 0, 0, 1, 0, 0])
#aqc.draw('mpl')
#qaoa = qc.assign_parameters([beta, gamma])
#qaoa.draw('mpl')
#plot_energy(np.load('./ws-energies/3reg0/energies/164_energy.npy'), title=str(np.array([1, 0, 0, 0, 1, 0])), show=False)
plot_energy(energy, title=str(np.array([0, 0, 0, 0, 1, 0])))

plt.show()
