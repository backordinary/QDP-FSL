# https://github.com/obstjn/ws-qaoa-transfer/blob/841b427f1c71a4005db74d482bce1729fcbd8f7a/scripts/landscape_3reg1_and_graph.py
import numpy as np
import networkx as nx
from networkx import Graph
import matplotlib.pyplot as plt
from qiskit import Aer, execute
from qiskit.visualization import plot_histogram

from plotting import *
from calculations import get_energy_grid, maximizing_parameters
from circuit_generation import qaoa_circuit


G3reg1 = Graph()
G3reg1.add_edges_from([(0,1), (1,2), (1,3), (2,3), (3,4)])
apx = np.array([1., .5, 1., 0, 0])

# draw ws graph
#nx.draw_kamada_kawai(G3reg1)
#draw_graph_with_ws(G3reg1, warmstarting=apx, show=False)
#plt.savefig('3reg1_graph.pdf')

# generate circuit
qc = qaoa_circuit(G3reg1)#, apx_sol=apx, eps=0.1)

# draw circuit
#qc.draw('mpl', filename='3reg1_circuit.pdf')

# calculate & save energy
#energy = get_energy_grid(G3reg1, qc, edge=(1,3), samples=100)
#np.save('./ws-energies/3-reg_1/23_energy.npy', energy)

# load energy
energy = np.load('./ws-energies/3-reg_1/energy_standard.npy')

# plot ws landscape
#plot_energy(energy, title=str(apx), filename='./ws-energies/3-reg_1/3reg1_23')

# plot histogram
#gamma_opt, beta_opt = maximizing_parameters(energy)
#gamma_opt, beta_opt = gamma_opt[0], beta_opt[0]
#print(gamma_opt, beta_opt)
#qaoa_instance = qc.assign_parameters([beta_opt, gamma_opt])
#result = execute(qaoa_instance, Aer.get_backend('statevector_simulator'), shots=1).result()
#plot_histogram(result.get_counts())
#plt.show()

# plot landscape with max parameter as marker
plot_energy_with_marker(energy, filename='3reg1_standard_marker')
